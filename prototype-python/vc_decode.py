"""Photo / image d'un code -> fichier audio.

  python vc_decode.py photo.jpg -o message.wav
"""
import argparse
import sys
import time

import cv2
import numpy as np

from vocalcode.detect import DecodeError, decode_image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="message.wav")
    a = ap.parse_args()

    img = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("image illisible", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    try:
        r = decode_image(img)
    except DecodeError as e:
        print(f"échec : {e}", file=sys.stderr)
        return 1
    dt = time.perf_counter() - t0
    worst = max(r.block_errors) / r.nsym
    print(f"grille {r.n}x{r.n}, repères {r.markers_found}/{r.markers_total}, "
          f"correction utilisée (pire bloc) {worst:.0%}, "
          f"{len(r.body)} octets en {dt * 1000:.0f} ms")

    import soundfile as sf
    from vocalcode import audio
    sf.write(a.out, audio.decode(r.body), audio.SR)
    print(f"-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
