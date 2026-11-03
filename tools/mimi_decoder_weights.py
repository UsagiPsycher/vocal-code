"""Extrait les poids du seul décodeur Mimi (8 codebooks) : float16, sauf le quantificateur (float32).

  python tools/mimi_decoder_weights.py <mimi complet .safetensors> <sortie .safetensors>
"""
import re
import sys
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

N_Q = 8
src, dst = sys.argv[1], sys.argv[2]
out = {}
with safe_open(src, "np") as f:
    for k in f.keys():
        if not k.startswith(("decoder.", "decoder_transformer.", "upsample.", "quantizer.")):
            continue
        m = re.match(r"quantizer\.rvq_rest\.vq\.layers\.(\d+)\.", k)
        if m and int(m.group(1)) >= N_Q - 1:
            continue
        t = f.get_tensor(k)
        if not k.startswith("quantizer.") and t.dtype == np.float32:
            if np.abs(t).max() > 6e4:
                raise SystemExit(f"{k} dépasse la plage float16")
            t = t.astype(np.float16)
        out[k] = np.ascontiguousarray(t)
save_file(out, dst)
print(f"{len(out)} tenseurs, {sum(v.nbytes for v in out.values()) / 1e6:.1f} Mo -> {dst}")
