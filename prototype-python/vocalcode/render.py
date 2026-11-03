"""Charge utile -> grille de modules -> image."""
import numpy as np
from PIL import Image

from .layout import (QUIET, FORMAT_SIDE, MIN_N, MAX_N, build_layout, whitening, slot_permutation)
from .ecc import block_plan, encode_format, pack_payload, rs_encode


def choose_size(payload_len: int, level: int) -> int:
    for n in range(MIN_N, MAX_N + 1):
        if block_plan(build_layout(n).n_slots, level).data_capacity >= payload_len:
            return n
    raise ValueError("message trop long")


def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, np.uint8))


def build_grid(body: bytes, level: int = 2, n: int | None = None) -> np.ndarray:
    payload = pack_payload(body)
    n = n or choose_size(len(payload), level)
    lay = build_layout(n)
    plan = block_plan(lay.n_slots, level)
    if len(payload) > plan.data_capacity:
        raise ValueError("charge utile trop grande pour cette taille")
    stream = rs_encode(payload + bytes(plan.data_capacity - len(payload)), plan)

    # octet i du flux -> emplacement perm[i] ; emplacement s = modules de données [8s, 8s+8)
    perm = slot_permutation(lay.n_slots, n)
    bits = np.zeros(len(lay.data_xy), np.uint8)
    sb = bytes_to_bits(stream).reshape(-1, 8)
    idx = (perm[:, None] * 8 + np.arange(8)).reshape(-1)
    bits[idx] = sb.reshape(-1)
    bits ^= whitening(len(bits))

    grid = lay.fixed.copy()
    grid[lay.data_xy[:, 1], lay.data_xy[:, 0]] = bits
    fbits = bytes_to_bits(encode_format(n, level)).reshape(FORMAT_SIDE, FORMAT_SIDE)
    for x, y in lay.format_areas:
        grid[y:y + FORMAT_SIDE, x:x + FORMAT_SIDE] = fbits
    return grid


def grid_to_image(grid: np.ndarray, px_per_module: int = 10) -> Image.Image:
    g = np.pad(grid, QUIET)
    img = np.where(g == 1, 0, 255).astype(np.uint8)
    img = np.kron(img, np.ones((px_per_module, px_per_module), np.uint8))
    return Image.fromarray(img, "L")
