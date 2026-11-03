"""Audio -> image du code, prête à imprimer.

  python vc_encode.py message.mp3 -o message.png --largeur-cm 8
"""
import argparse
import sys
from pathlib import Path

from vocalcode import audio
from vocalcode.layout import QUIET
from vocalcode.render import build_grid, grid_to_image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", help="image de sortie (.png)")
    ap.add_argument("--max-secondes", type=float, default=30)
    ap.add_argument("--ecc", type=int, default=2, choices=[0, 1, 2, 3],
                    help="niveau de correction : 0=20%%, 1=30%%, 2=40%% (défaut), 3=50%% de parité")
    ap.add_argument("--largeur-cm", type=float, default=8, help="largeur imprimée, zone blanche incluse")
    ap.add_argument("--px-module", type=int, default=12)
    a = ap.parse_args()

    out = Path(a.out or Path(a.audio).with_suffix(".png"))
    wav = audio.load_audio(a.audio, a.max_secondes)
    body = audio.encode(wav)
    out.with_suffix(".bin").write_bytes(body)
    grid = build_grid(body, a.ecc)
    n = grid.shape[0]
    img = grid_to_image(grid, a.px_module)
    dpi = img.width / (a.largeur_cm / 2.54)
    img.save(out, dpi=(dpi, dpi))
    module_mm = a.largeur_cm * 10 / (n + 2 * QUIET)
    print(f"{len(wav) / audio.SR:.1f} s -> {len(body)} octets audio, grille {n}x{n}, "
          f"module {module_mm:.2f} mm à {a.largeur_cm} cm -> {out}")


if __name__ == "__main__":
    sys.exit(main())
