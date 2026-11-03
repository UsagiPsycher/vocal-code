"""Encode -> fichier binaire compact (11 bits/code) -> relecture du fichier seul -> décodage."""
import sys
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import MimiModel

src, binpath, outwav = sys.argv[1], sys.argv[2], sys.argv[3]
N = 8
model = MimiModel.from_pretrained("kyutai/mimi").eval()

# --- Encodage ---
wav, sr = sf.read(src)
if wav.ndim > 1:
    wav = wav.mean(1)
wav = resample_poly(wav, 24000, sr).astype(np.float32)
with torch.no_grad():
    codes = model.encode(torch.from_numpy(wav)[None, None], num_quantizers=N).audio_codes[0]  # (N, T)
T = codes.shape[1]
vals = codes.T.reshape(-1).numpy().astype(np.uint16)  # ordre temporel : trame par trame
bits = ((vals[:, None] >> np.arange(10, -1, -1)) & 1).astype(np.uint8).reshape(-1)
payload = T.to_bytes(2, "big") + np.packbits(bits).tobytes()  # en-tête : nb de trames (2 octets)
open(binpath, "wb").write(payload)
print(f"fichier compressé : {len(payload)} octets ({T} trames x {N} codebooks)")

# --- Décodage à partir du fichier uniquement ---
data = open(binpath, "rb").read()
T2 = int.from_bytes(data[:2], "big")
b = np.unpackbits(np.frombuffer(data[2:], np.uint8))[: T2 * N * 11].reshape(-1, 11)
v = (b.astype(np.uint16) << np.arange(10, -1, -1)).sum(1)
codes2 = torch.from_numpy(v.reshape(T2, N).T.astype(np.int64))[None]
assert torch.equal(codes2[0], codes), "les codes relus diffèrent"
with torch.no_grad():
    out = model.decode(codes2).audio_values[0, 0].numpy()
sf.write(outwav, out, 24000)
print("décodage depuis le fichier : OK")
