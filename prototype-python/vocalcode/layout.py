"""Disposition de la grille : repères, zones de format, modules de données.

Conventions : grille N x N modules, (x = colonne, y = ligne), 1 = noir.
Le centre du module (x, y) a pour coordonnées continues (x + 0.5, y + 0.5).
"""
from dataclasses import dataclass, field
from functools import lru_cache
import math
import numpy as np

QUIET = 4            # zone blanche autour du code (en modules)
FINDER = 7           # repère de coin 7x7
MARKER = 5           # repère d'alignement 5x5
MARKER_SPACING = 20  # espacement visé entre repères d'alignement
FORMAT_SIDE = 8      # zone de format 8x8 = 64 bits
MIN_N, MAX_N = 41, 1023

KIND_DATA, KIND_FINDER, KIND_FORMAT, KIND_MARKER = 0, 1, 2, 3


def xorshift32(seed: int):
    x = seed & 0xFFFFFFFF or 1
    while True:
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        yield x


def finder_pattern() -> np.ndarray:
    p = np.ones((FINDER, FINDER), np.uint8)
    p[1:6, 1:6] = 0
    p[2:5, 2:5] = 1
    return p


def marker_pattern() -> np.ndarray:
    p = np.ones((MARKER, MARKER), np.uint8)
    p[1:4, 1:4] = 0
    p[2, 2] = 1
    return p


@dataclass
class Layout:
    n: int
    kind: np.ndarray                 # (N, N) type de chaque module
    fixed: np.ndarray                # (N, N) valeur des modules non-données
    finder_origins: list             # [(x, y)] coin haut-gauche de TL, TR, BR, BL (sens horaire)
    marker_centers: list             # [(cx, cy)] indices du module central
    format_areas: list               # [(x0, y0)] coin haut-gauche des zones 8x8
    data_xy: np.ndarray = field(repr=False)  # (M, 2) modules de données, ordre ligne par ligne

    @property
    def finder_centers(self):
        return [(x + 3.5, y + 3.5) for x, y in self.finder_origins]

    @property
    def n_slots(self):
        return len(self.data_xy) // 8


def _marker_axis(n: int) -> list:
    k = max(2, math.ceil((n - 5) / MARKER_SPACING) + 1)
    return [int(round(v)) for v in np.linspace(2, n - 3, k)]


@lru_cache(maxsize=64)
def build_layout(n: int) -> Layout:
    if not MIN_N <= n <= MAX_N:
        raise ValueError(f"taille de grille hors limites : {n}")
    kind = np.zeros((n, n), np.uint8)
    fixed = np.zeros((n, n), np.uint8)

    origins = [(0, 0), (n - FINDER, 0), (n - FINDER, n - FINDER), (0, n - FINDER)]
    fp = finder_pattern()
    for ox, oy in origins:
        # séparateur blanc de 1 module autour du repère
        x0, x1 = max(ox - 1, 0), min(ox + FINDER + 1, n)
        y0, y1 = max(oy - 1, 0), min(oy + FINDER + 1, n)
        kind[y0:y1, x0:x1] = KIND_FINDER
        fixed[y0:y1, x0:x1] = 0
        fixed[oy:oy + FINDER, ox:ox + FINDER] = fp

    s = FORMAT_SIDE
    areas = [(9, 0), (0, 9), (n - s, 9), (9, n - s)]  # droite TL, sous TL, sous TR, droite BL
    for x, y in areas:
        kind[y:y + s, x:x + s] = KIND_FORMAT

    mp = marker_pattern()
    centers = []
    axis = _marker_axis(n)
    for cy in axis:
        for cx in axis:
            sl = (slice(cy - 2, cy + 3), slice(cx - 2, cx + 3))
            if np.any(kind[sl] != KIND_DATA):
                continue
            kind[sl] = KIND_MARKER
            fixed[sl] = mp
            centers.append((cx, cy))

    ys, xs = np.nonzero(kind == KIND_DATA)  # np.nonzero parcourt ligne par ligne
    data_xy = np.stack([xs, ys], 1)
    return Layout(n, kind, fixed, origins, centers, areas, data_xy)


@lru_cache(maxsize=64)
def whitening(n_bits: int) -> np.ndarray:
    """Masque pseudo-aléatoire XOR appliqué aux modules de données."""
    g = xorshift32(0x6C8E9CF5)
    return np.array([next(g) >> 31 for _ in range(n_bits)], np.uint8)


@lru_cache(maxsize=64)
def slot_permutation(n_slots: int, n: int) -> np.ndarray:
    """perm[i] = emplacement (groupe de 8 modules) qui reçoit l'octet i du flux."""
    perm = list(range(n_slots))
    g = xorshift32(0x9E3779B9 ^ n)
    for i in range(n_slots - 1, 0, -1):
        j = next(g) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return np.array(perm, np.int64)
