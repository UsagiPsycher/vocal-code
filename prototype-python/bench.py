"""Banc de robustesse : abîme l'image du code de façon contrôlée et tente de la décoder.

  python bench.py echantillons/_brut/mimi_1100bps.bin [--essais 8] [--ecc 2] [--exemples dossier]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from vocalcode.detect import DecodeError, decode_image
from vocalcode.render import build_grid, grid_to_image

HI = 8  # px/module du rendu de travail, avant réduction


def cylinder(img, arc_deg, rng):
    """Code collé sur un cylindre (poteau) : le code couvre arc_deg degrés de l'arc."""
    if arc_deg <= 0:
        return img
    h, w = img.shape
    phi = np.radians(arc_deg)
    R = w / phi
    xs = np.arange(w) - w / 2
    out_half = R * np.sin(min(phi / 2, np.pi / 2 - 1e-3))
    xo = np.linspace(-out_half, out_half, w)
    src_x = R * np.arcsin(np.clip(xo / R, -1, 1)) + w / 2
    shade = 0.55 + 0.45 * np.cos(np.arcsin(np.clip(xo / R, -1, 1)))
    mx = np.tile(src_x.astype(np.float32), (h, 1))
    my = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=230)
    return out * shade[None, :].astype(np.float32)


def perspective(img, amount, rng):
    if amount <= 0:
        return img
    h, w = img.shape
    pad = int(amount * max(h, w))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=230)
    H2, W2 = img.shape
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    dst = src + rng.uniform(-amount, amount, (4, 2)).astype(np.float32) * np.float32([w, h])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W2, H2), borderValue=230)


def lighting(img, strength, glare, rng):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if strength > 0:
        ang = rng.uniform(0, 2 * np.pi)
        g = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        img = img * (1 - strength * g)
    if glare > 0:
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        r = glare * min(h, w)
        blob = np.exp(-(((xx - cx) / r) ** 2 + ((yy - cy) / (0.6 * r)) ** 2))
        img = img + blob * 255 * 0.9
    return img


def stains(img, frac, rng, code_box):
    if frac <= 0:
        return img
    x0, y0, x1, y1 = code_box
    area = (x1 - x0) * (y1 - y0)
    covered = 0
    mask = np.zeros(img.shape, np.uint8)
    while covered < frac * area:
        r = int(rng.uniform(0.02, 0.06) * (x1 - x0))
        c = (int(rng.uniform(x0, x1)), int(rng.uniform(y0, y1)))
        cv2.ellipse(mask, c, (r, int(r * rng.uniform(0.4, 1))), rng.uniform(0, 180), 0, 360, 1, -1)
        covered = mask[y0:y1, x0:x1].sum()
    tone = np.where(rng.random(img.shape) < 0.5, 40, 220).astype(np.float32)
    tone = cv2.GaussianBlur(tone, (0, 0), 6)
    return np.where(mask == 1, tone, img)


def degrade(base, rng, *, ppm=4.0, blur=0.0, persp=0.0, arc=0.0, light=0.0, glare=0.0,
            stain=0.0, noise=2.0, jpeg=90):
    img = base.astype(np.float32) * (175 / 255) + 45  # encre pas tout à fait noire, papier pas tout à fait blanc
    h, w = img.shape
    margin = HI * 6
    img = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=230)
    img = stains(img, stain, rng, (margin, margin, margin + w, margin + h))
    img = cylinder(img, arc, rng)
    img = perspective(img, persp, rng)
    img = lighting(img, light, glare, rng)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur * HI)
    s = ppm / HI
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


SCENARIOS = [
    ("propre, 4 px/module", {}),
    ("3 px/module", dict(ppm=3)),
    ("2,5 px/module", dict(ppm=2.5)),
    ("flou 0,4 module", dict(blur=0.4)),
    ("flou 0,6 module", dict(blur=0.6)),
    ("perspective 15 %", dict(persp=0.15)),
    ("perspective 25 %", dict(persp=0.25)),
    ("poteau, arc 60°", dict(arc=60)),
    ("poteau, arc 100°", dict(arc=100)),
    ("éclairage inégal + reflet", dict(light=0.6, glare=0.12)),
    ("taches 5 %", dict(stain=0.05)),
    ("taches 10 %", dict(stain=0.10)),
    ("bruit fort + JPEG 50", dict(noise=10, jpeg=50)),
    ("réaliste combiné", dict(ppm=4, blur=0.3, persp=0.1, arc=45, light=0.4, glare=0.08,
                              stain=0.02, noise=5, jpeg=75)),
    ("dur combiné", dict(ppm=3, blur=0.4, persp=0.18, arc=70, light=0.6, glare=0.12,
                         stain=0.05, noise=8, jpeg=60)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--essais", type=int, default=8)
    ap.add_argument("--ecc", type=int, default=2)
    ap.add_argument("--exemples", help="dossier où enregistrer une image dégradée par scénario")
    ap.add_argument("--seul", help="ne lancer que les scénarios contenant ce texte")
    a = ap.parse_args()

    body = Path(a.bin).read_bytes()
    grid = build_grid(body, a.ecc)
    base = np.array(grid_to_image(grid, HI))
    print(f"grille {grid.shape[0]}x{grid.shape[0]}, ecc {a.ecc}, {len(body)} octets\n")
    print(f"{'scénario':28} {'réussite':>9} {'marge utilisée (médiane / max)':>32} {'temps':>7}")
    ex = Path(a.exemples) if a.exemples else None
    if ex:
        ex.mkdir(parents=True, exist_ok=True)
    for i, (name, kw) in enumerate(SCENARIOS):
        if a.seul and a.seul not in name:
            continue
        ok, used, times, errs = 0, [], [], []
        for k in range(a.essais):
            rng = np.random.default_rng(1000 * i + k)
            img = degrade(base, rng, **kw)
            if ex and k == 0:
                cv2.imwrite(str(ex / f"{i:02d}.jpg"), img)
            t0 = time.perf_counter()
            try:
                r = decode_image(img)
                if r.body == body:
                    ok += 1
                    used.append(max(r.block_errors) / r.nsym)
                else:
                    errs.append("contenu différent")
            except DecodeError as e:
                errs.append(str(e).split(" ;")[0])
            times.append(time.perf_counter() - t0)
        marge = f"{np.median(used):.0%} / {max(used):.0%}" if used else "-"
        print(f"{name:28} {ok:>4}/{a.essais:<4} {marge:>32} {np.median(times) * 1000:>5.0f}ms"
              + (f"   [{errs[0]}]" if errs else ""))


if __name__ == "__main__":
    main()
