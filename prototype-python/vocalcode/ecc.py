"""Correction d'erreurs Reed-Solomon (GF(256), polynôme 0x11d, fcr=0) et en-têtes."""
from dataclasses import dataclass
import math
import struct
import zlib
import numpy as np
from reedsolo import RSCodec, ReedSolomonError

FORMAT_VERSION = 1
ECC_RATIOS = {0: 0.20, 1: 0.30, 2: 0.40, 3: 0.50}  # part de parité par bloc
FORMAT_NSYM = 6
FORMAT_MASK = bytes([0xA5, 0x3C, 0x96, 0x5A, 0xC3, 0x69, 0x0F, 0xF0])
CODEC_MIMI8 = 1

_codecs = {}


def _rs(nsym: int) -> RSCodec:
    if nsym not in _codecs:
        _codecs[nsym] = RSCodec(nsym)
    return _codecs[nsym]


# ---------- Bloc de format (64 bits, répété 4 fois) ----------

def encode_format(n: int, level: int) -> bytes:
    info = (n << 6) | (level << 4) | FORMAT_VERSION  # 10 + 2 + 4 bits
    cw = bytes(_rs(FORMAT_NSYM).encode(struct.pack(">H", info)))
    return bytes(a ^ b for a, b in zip(cw, FORMAT_MASK))


def decode_format(cw: bytes):
    raw = bytearray(a ^ b for a, b in zip(cw, FORMAT_MASK))
    try:
        msg = _rs(FORMAT_NSYM).decode(raw)[0]
    except ReedSolomonError:
        return None
    info = struct.unpack(">H", bytes(msg))[0]
    if info & 0xF != FORMAT_VERSION:
        return None
    return info >> 6, (info >> 4) & 0x3


# ---------- Découpage en blocs ----------

@dataclass
class BlockPlan:
    sizes: list   # taille totale de chaque bloc (<= 255)
    nsym: int     # octets de parité par bloc

    @property
    def data_capacity(self):
        return sum(s - self.nsym for s in self.sizes)


def block_plan(n_slots: int, level: int) -> BlockPlan:
    nblocks = max(1, math.ceil(n_slots / 255))
    base, extra = divmod(n_slots, nblocks)
    sizes = [base + 1] * extra + [base] * (nblocks - extra)
    nsym = max(2, int(round(base * ECC_RATIOS[level])))
    return BlockPlan(sizes, nsym)


def rs_encode(data: bytes, plan: BlockPlan) -> bytes:
    assert len(data) == plan.data_capacity
    out, pos, rs = bytearray(), 0, _rs(plan.nsym)
    for s in plan.sizes:
        k = s - plan.nsym
        out += rs.encode(data[pos:pos + k])
        pos += k
    return bytes(out)


def rs_decode(stream: bytes, conf: np.ndarray, plan: BlockPlan):
    """Décode chaque bloc ; en cas d'échec, réessaie en effaçant les octets les moins sûrs.

    Renvoie (données, coût par bloc en symboles de parité) ou lève ReedSolomonError.
    """
    rs = _rs(plan.nsym)
    out, pos, stats = bytearray(), 0, []
    for s in plan.sizes:
        block = bytearray(stream[pos:pos + s])
        c = conf[pos:pos + s]
        pos += s
        attempts = [0] + [int(plan.nsym * f) for f in (0.5, 0.75, 0.95)]
        for n_erase in attempts:
            erase = sorted(np.argsort(c)[:n_erase].tolist()) if n_erase else None
            try:
                msg, _, errata = rs.decode(block, erase_pos=erase)
                # coût en symboles de parité : 1 par effacement, 2 par erreur
                stats.append(n_erase + 2 * max(0, len(errata) - n_erase))
                break
            except ReedSolomonError:
                continue
        else:
            raise ReedSolomonError("bloc irrécupérable")
        out += msg
    return bytes(out), stats


# ---------- Charge utile ----------

def pack_payload(body: bytes, codec: int = CODEC_MIMI8) -> bytes:
    head = struct.pack(">BBH", FORMAT_VERSION, codec, len(body))
    return head + body + struct.pack(">I", zlib.crc32(head + body))


def unpack_payload(data: bytes):
    ver, codec, length = struct.unpack(">BBH", data[:4])
    body = data[4:4 + length]
    crc = struct.unpack(">I", data[4 + length:8 + length])[0]
    if ver != FORMAT_VERSION or zlib.crc32(data[:4 + length]) != crc:
        raise ValueError("CRC invalide")
    return codec, body
