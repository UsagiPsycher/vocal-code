"""Image (photo) -> charge utile.

Étapes :
1. repérer les 3 repères de coin (carrés concentriques) ;
2. lire le bloc de format (taille de grille, niveau de correction) ;
3. recaler progressivement chaque repère d'alignement, du coin vers le reste
   (gère perspective et courbure) ;
4. interpoler la position de chaque module (homographie globale + spline de plaque mince) ;
5. échantillonner, seuiller localement, démêler, corriger les erreurs, vérifier le CRC.
"""
from dataclasses import dataclass
import itertools
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import convolve, uniform_filter

from .layout import (FORMAT_SIDE, KIND_DATA, MIN_N, MAX_N, build_layout, marker_pattern,
                     slot_permutation, whitening)
from .ecc import block_plan, decode_format, rs_decode, unpack_payload


EQ_KERNEL = np.array([[0, 0.2, 0], [0.2, 0, 0.2], [0, 0.2, 0]])


class DecodeError(Exception):
    pass


@dataclass
class Finder:
    center: np.ndarray
    corners: np.ndarray   # 4 x 2, non ordonnés
    module: float         # taille estimée d'un module (px)


# ---------- 1. Repères de coin ----------

def _finder_candidates(gray: np.ndarray) -> list:
    h, w = gray.shape
    found = []
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 2.5, blur, -1.5, 0)
    variants = [(g, frac) for g in (gray, sharp) for frac in (0.02, 0.05, 0.12)]
    for src, frac in variants:
        bs = max(11, int(min(h, w) * frac) | 1)
        binv = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, bs, 5)
        contours, hier = cv2.findContours(binv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        hier = hier[0]
        for i, c in enumerate(contours):
            ch = hier[i][2]
            if ch < 0:
                continue
            gch = hier[ch][2]
            if gch < 0:
                continue
            a0 = cv2.contourArea(c)
            a1 = cv2.contourArea(contours[ch])
            a2 = cv2.contourArea(contours[gch])
            if a0 < 49 or a1 <= 0 or a2 <= 0:
                continue
            # théorie : 49 : 25 : 9
            if not (1.4 < a0 / a1 < 3.0 and 3.0 < a0 / a2 < 10.0):
                continue
            hull = cv2.convexHull(c)
            if a0 / max(cv2.contourArea(hull), 1) < 0.85:
                continue
            m = cv2.moments(contours[gch])
            if m["m00"] == 0:
                continue
            center = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            approx = cv2.approxPolyDP(hull, 0.08 * cv2.arcLength(hull, True), True).reshape(-1, 2)
            corners = approx.astype(float) if len(approx) == 4 else None
            found.append(Finder(center, corners, np.sqrt(a0) / 7.0))
    # dédoublonnage
    uniq = []
    for f in sorted(found, key=lambda f: -f.module):
        if all(np.linalg.norm(f.center - u.center) > 2 * u.module for u in uniq):
            uniq.append(f)
    return uniq


def _pick_triple(cands: list):
    best, best_score = None, 1e9
    for tri in itertools.combinations(cands[:12], 3):
        mods = np.array([f.module for f in tri])
        if mods.max() / mods.min() > 3.0:
            continue
        for k in range(3):
            tl = tri[k]  # sommet de l'angle droit
            a, b = [tri[j] for j in range(3) if j != k]
            va, vb = a.center - tl.center, b.center - tl.center
            la, lb = np.linalg.norm(va), np.linalg.norm(vb)
            if min(la, lb) < 10 * mods.mean():
                continue
            cos = abs(va @ vb) / (la * lb)
            ratio = max(la, lb) / min(la, lb)
            if cos > 0.7 or ratio > 3.0:
                continue
            score = cos + (ratio - 1) * 0.5 + (mods.max() / mods.min() - 1) * 0.5
            if score < best_score:
                cross = va[0] * vb[1] - va[1] * vb[0]
                nxt, prev = (a, b) if cross > 0 else (b, a)
                best, best_score = (prev, tl, nxt), score
    return best  # (précédent, sommet, suivant) dans le sens horaire


# ---------- outils géométriques ----------

def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _fit_map(mod: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Homographie module -> image (3x3), ou affine si trop peu de points."""
    if len(mod) >= 4:
        H, _ = cv2.findHomography(mod.astype(np.float64), img.astype(np.float64), 0)
        if H is not None:
            return H
    A = cv2.getAffineTransform(mod[:3].astype(np.float32), img[:3].astype(np.float32))
    return np.vstack([A, [0, 0, 1]])


def _corner_centers(n: int):
    return [(3.5, 3.5), (n - 3.5, 3.5), (n - 3.5, n - 3.5), (3.5, n - 3.5)]  # sens horaire


def _assign(triple, n: int, rot: int):
    """Hypothèse d'orientation : le sommet de l'angle droit est le coin n° rot."""
    cc = _corner_centers(n)
    prev, v, nxt = triple
    return [(prev, cc[(rot - 1) % 4]), (v, cc[rot]), (nxt, cc[(rot + 1) % 4])]


def _finder_anchors(pairs):
    """Centres (+ coins si détectés) des repères de coin, en coordonnées module et image."""
    mod = [c for _, c in pairs]
    img = [f.center for f, _ in pairs]
    H = _fit_map(np.array(mod), np.array(img))
    for f, (cx, cy) in pairs:
        if f.corners is None:
            continue
        ideal = np.array([(cx - 3.5, cy - 3.5), (cx + 3.5, cy - 3.5),
                          (cx + 3.5, cy + 3.5), (cx - 3.5, cy + 3.5)])
        pred = _apply_h(H, ideal)
        for m_pt, p in zip(ideal, pred):
            d = np.linalg.norm(f.corners - p, axis=1)
            if d.min() < 2.5 * f.module:
                mod.append(tuple(m_pt))
                img.append(f.corners[d.argmin()])
    return np.array(mod, float), np.array(img, float)


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    cols = 1024
    rows = -(-n // cols)
    buf = np.zeros((rows * cols, 2), np.float32)
    buf[:n] = pts
    mx = np.ascontiguousarray(buf[:, 0].reshape(rows, cols))
    my = np.ascontiguousarray(buf[:, 1].reshape(rows, cols))
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1)[:n]


# ---------- 2. Bloc de format ----------

def _read_format(contrast: np.ndarray, triple, n_est: float):
    """Essaie chaque orientation et chaque taille plausible ; le bloc de format tranche."""
    s = FORMAT_SIDE
    offs = np.array([(i % s + 0.5, i // s + 0.5) for i in range(s * s)])
    lo, hi = max(MIN_N, int(n_est * 0.6)), min(MAX_N, int(n_est * 1.4) + 1)
    for n in sorted(range(lo, hi), key=lambda v: abs(v - n_est)):
        lay = build_layout(n)
        for rot in range(4):
            pairs = _assign(triple, n, rot)
            mod, img = _finder_anchors(pairs)
            maps = [_fit_map(mod[:3], img[:3])] + ([_fit_map(mod, img)] if len(mod) >= 6 else [])
            for H in maps:
                per_copy = [_sample(contrast, _apply_h(H, offs + (x, y))) for x, y in lay.format_areas]
                # combinaison souple des 4 copies, puis chaque copie seule
                for soft in [np.sum(per_copy, 0)] + per_copy:
                    bits = (soft < 0).astype(np.uint8)
                    res = decode_format(np.packbits(bits).tobytes())
                    if res and MIN_N <= res[0] <= MAX_N and abs(res[0] - n_est) < 0.45 * n_est:
                        if res[0] != n:  # taille lue différente : on réassigne avec la bonne
                            pairs = _assign(triple, res[0], rot)
                        return res[0], res[1], pairs
    return None


# ---------- 3. Recalage des repères d'alignement ----------

P = 8  # pixels par module dans les vignettes redressées
_TEMPLATE = np.kron(np.where(marker_pattern() == 1, 0, 255).astype(np.float32),
                    np.ones((P, P), np.float32))


def _refine_marker(gray_f: np.ndarray, H: np.ndarray, cx: int, cy: int):
    half = 6.5
    ox, oy = cx + 0.5 - half, cy + 0.5 - half
    T = np.array([[1 / P, 0, ox + 0.5 / P], [0, 1 / P, oy + 0.5 / P], [0, 0, 1]])
    size = int(2 * half * P)
    patch = cv2.warpPerspective(gray_f, H @ T, (size, size),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
    res = cv2.matchTemplate(patch, _TEMPLATE, cv2.TM_CCOEFF_NORMED)
    _, score, _, (u, v) = cv2.minMaxLoc(res)
    du = dv = 0.0
    if 0 < u < res.shape[1] - 1:
        l, c, r = res[v, u - 1], res[v, u], res[v, u + 1]
        du = 0.5 * (l - r) / (l - 2 * c + r + 1e-9)
    if 0 < v < res.shape[0] - 1:
        t, c, b = res[v - 1, u], res[v, u], res[v + 1, u]
        dv = 0.5 * (t - b) / (t - 2 * c + b + 1e-9)
    half_t = _TEMPLATE.shape[0] / 2
    m = np.array([[ox + (u + du + half_t) / P, oy + (v + dv + half_t) / P]])
    return score, _apply_h(H, m)[0]


def _locate_markers(gray_f, lay, mod, img, found_centers, k=8, min_score=0.45):
    mod, img = list(map(tuple, mod)), list(map(tuple, img))
    fc = np.array(found_centers)
    order = sorted(lay.marker_centers,
                   key=lambda c: np.min(np.linalg.norm(fc - (c[0] + 0.5, c[1] + 0.5), axis=1)))
    n_ok = 0
    for cx, cy in order:
        m_arr, i_arr = np.array(mod), np.array(img)
        d = np.linalg.norm(m_arr - (cx + 0.5, cy + 0.5), axis=1)
        near = np.argsort(d)[:k]
        H = _fit_map(m_arr[near], i_arr[near])
        score, pos = _refine_marker(gray_f, H, cx, cy)
        if score >= min_score:
            mod.append((cx + 0.5, cy + 0.5))
            img.append(tuple(pos))
            n_ok += 1
    return np.array(mod), np.array(img), n_ok


# ---------- 4-5. Échantillonnage et décodage ----------

def _module_positions(lay, mod, img) -> np.ndarray:
    H, _ = cv2.findHomography(mod, img, 0)
    resid = img - _apply_h(H, mod)
    scale = np.median(np.linalg.norm(np.diff(_apply_h(H, np.array([[0, 0], [1, 0]])), axis=0)))
    rbf = RBFInterpolator(mod, resid, kernel="thin_plate_spline", smoothing=0.5 * scale)
    n = lay.n
    ys, xs = np.mgrid[0:n, 0:n]
    centers = np.c_[xs.ravel() + 0.5, ys.ravel() + 0.5]
    return (_apply_h(H, centers) + rbf(centers)).reshape(n, n, 2), scale


@dataclass
class DecodeResult:
    codec: int
    body: bytes
    n: int
    level: int
    markers_found: int
    markers_total: int
    block_errors: list
    nsym: int


def decode_image(image: np.ndarray) -> DecodeResult:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cands = _finder_candidates(gray)
    triple = _pick_triple(cands)
    if triple is None:
        raise DecodeError(f"repères de coin introuvables ({len(cands)} candidats)")
    prev, v, nxt = triple
    mod_px = np.mean([f.module for f in triple])
    dist = (np.linalg.norm(nxt.center - v.center) + np.linalg.norm(prev.center - v.center)) / 2
    n_est = dist / mod_px + 7

    gray_f = gray.astype(np.float32)
    contrast = gray_f - cv2.blur(gray_f, (int(mod_px * 9) | 1,) * 2)

    fmt = _read_format(contrast, triple, n_est)
    if fmt is None:
        raise DecodeError("bloc de format illisible")
    n, level, pairs = fmt
    lay = build_layout(n)

    # 4e repère de coin, s'il est visible
    used = {id(f) for f, _ in pairs}
    mod, img = _finder_anchors(pairs)
    H = _fit_map(mod, img)
    for c in _corner_centers(n):
        if c in [m for _, m in pairs]:
            continue
        p = _apply_h(H, np.array([c]))[0]
        near = [f for f in cands if id(f) not in used and np.linalg.norm(f.center - p) < 4 * mod_px]
        if near:
            pairs.append((min(near, key=lambda f: np.linalg.norm(f.center - p)), c))

    mod, img = _finder_anchors(pairs)
    mod, img, n_ok = _locate_markers(gray_f, lay, mod, img, [m for _, m in pairs])
    pos, _ = _module_positions(lay, mod, img)

    vals = _sample(contrast, pos.reshape(-1, 2)).reshape(n, n)
    # égaliseur : retire la part de signal « bavée » par les 4 voisins (flou)
    vals = vals - convolve(vals, EQ_KERNEL, mode="nearest")
    data_mask = (lay.kind == KIND_DATA).astype(np.float32)
    w = 9
    local = uniform_filter(vals * data_mask, w) / np.maximum(uniform_filter(data_mask, w), 1e-6)
    centered = vals - local
    # normalisation sur une fenêtre large : une zone uniforme (tache) garde une confiance faible
    w2 = 31
    spread = np.sqrt(uniform_filter(centered ** 2 * data_mask, w2)
                     / np.maximum(uniform_filter(data_mask, w2), 1e-6)) + 1e-6
    xs, ys = lay.data_xy[:, 0], lay.data_xy[:, 1]
    soft = centered[ys, xs] / spread[ys, xs]

    bits = (soft < 0).astype(np.uint8) ^ whitening(len(soft))
    conf_mod = np.abs(soft)
    perm = slot_permutation(lay.n_slots, n)
    idx = perm[:, None] * 8 + np.arange(8)
    stream = np.packbits(bits[idx].reshape(-1)).tobytes()
    conf = conf_mod[idx].min(axis=1)

    plan = block_plan(lay.n_slots, level)
    try:
        data, stats = rs_decode(stream, conf, plan)
        codec, body = unpack_payload(data)
    except Exception as e:
        raise DecodeError(f"données irrécupérables ({e}) ; repères recalés {n_ok}/{len(lay.marker_centers)}")
    return DecodeResult(codec, body, n, level, n_ok, len(lay.marker_centers), stats, plan.nsym)
