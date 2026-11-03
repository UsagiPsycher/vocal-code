"""Audio <-> octets via le codec neuronal Mimi (Kyutai), 8 codebooks, ~1,1 kbps."""
import numpy as np

SR = 24000
N_Q = 8
BITS = 11          # 2048 entrées par codebook
FRAME_RATE = 12.5  # trames par seconde

_model = None


def _get_model():
    global _model
    if _model is None:
        from transformers import MimiModel
        _model = MimiModel.from_pretrained("kyutai/mimi").eval()
    return _model


def load_audio(path: str, max_seconds: float) -> np.ndarray:
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(max_seconds),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def encode(wav: np.ndarray) -> bytes:
    import torch
    with torch.no_grad():
        codes = _get_model().encode(torch.from_numpy(wav.copy())[None, None],
                                    num_quantizers=N_Q).audio_codes[0]
    t = codes.shape[1]
    vals = codes.T.reshape(-1).numpy().astype(np.uint16)
    bits = ((vals[:, None] >> np.arange(BITS - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
    return t.to_bytes(2, "big") + np.packbits(bits).tobytes()


def decode(body: bytes) -> np.ndarray:
    import torch
    t = int.from_bytes(body[:2], "big")
    b = np.unpackbits(np.frombuffer(body[2:], np.uint8))[: t * N_Q * BITS].reshape(-1, BITS)
    v = (b.astype(np.uint16) << np.arange(BITS - 1, -1, -1)).sum(1)
    codes = torch.from_numpy(v.reshape(t, N_Q).T.astype(np.int64))[None]
    with torch.no_grad():
        return _get_model().decode(codes).audio_values[0, 0].numpy()
