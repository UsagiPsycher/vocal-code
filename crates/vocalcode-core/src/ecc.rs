//! Bloc de format, découpage en blocs Reed-Solomon, charge utile.

use crate::rs;

pub const FORMAT_VERSION: u8 = 1;
pub const FORMAT_NSYM: usize = 6;
const FORMAT_MASK: [u8; 8] = [0xA5, 0x3C, 0x96, 0x5A, 0xC3, 0x69, 0x0F, 0xF0];
pub const CODEC_MIMI8: u8 = 1;

/// Part de parité par bloc selon le niveau de correction.
pub fn ecc_ratio(level: u8) -> f64 {
    [0.20, 0.30, 0.40, 0.50][level as usize]
}

// ---------- Bloc de format (64 bits, répété 4 fois) ----------

pub fn encode_format(n: usize, level: u8) -> [u8; 8] {
    let info = ((n as u16) << 6) | ((level as u16) << 4) | FORMAT_VERSION as u16;
    let cw = rs::encode(&info.to_be_bytes(), FORMAT_NSYM);
    let mut out = [0u8; 8];
    for i in 0..8 {
        out[i] = cw[i] ^ FORMAT_MASK[i];
    }
    out
}

pub fn decode_format(cw: &[u8; 8]) -> Option<(usize, u8)> {
    let raw: Vec<u8> = cw.iter().zip(FORMAT_MASK).map(|(a, b)| a ^ b).collect();
    let (msg, _) = rs::decode(&raw, FORMAT_NSYM, &[]).ok()?;
    let info = u16::from_be_bytes([msg[0], msg[1]]);
    if info & 0xF != FORMAT_VERSION as u16 {
        return None;
    }
    Some(((info >> 6) as usize, ((info >> 4) & 0x3) as u8))
}

// ---------- Découpage en blocs ----------

#[derive(Clone, Debug)]
pub struct BlockPlan {
    pub sizes: Vec<usize>,
    pub nsym: usize,
}

impl BlockPlan {
    pub fn data_capacity(&self) -> usize {
        self.sizes.iter().map(|s| s - self.nsym).sum()
    }
}

pub fn block_plan(n_slots: usize, level: u8) -> BlockPlan {
    let nblocks = n_slots.div_ceil(255).max(1);
    let (base, extra) = (n_slots / nblocks, n_slots % nblocks);
    let sizes = (0..nblocks).map(|i| base + (i < extra) as usize).collect();
    let nsym = ((base as f64 * ecc_ratio(level)).round() as usize).max(2);
    BlockPlan { sizes, nsym }
}

pub fn rs_encode(data: &[u8], plan: &BlockPlan) -> Vec<u8> {
    assert_eq!(data.len(), plan.data_capacity());
    let mut out = Vec::with_capacity(plan.sizes.iter().sum());
    let mut pos = 0;
    for &s in &plan.sizes {
        let k = s - plan.nsym;
        out.extend(rs::encode(&data[pos..pos + k], plan.nsym));
        pos += k;
    }
    out
}

/// Décode chaque bloc ; en cas d'échec, réessaie en effaçant les octets les moins sûrs.
/// Renvoie (données, coût par bloc en symboles de parité : 1 par effacement, 2 par erreur).
pub fn rs_decode(stream: &[u8], conf: &[f32], plan: &BlockPlan) -> Option<(Vec<u8>, Vec<usize>)> {
    let mut out = Vec::with_capacity(plan.data_capacity());
    let mut costs = Vec::new();
    let mut pos = 0;
    for &s in &plan.sizes {
        let block = &stream[pos..pos + s];
        let c = &conf[pos..pos + s];
        pos += s;
        let mut order: Vec<usize> = (0..s).collect();
        order.sort_by(|&a, &b| c[a].total_cmp(&c[b]));
        let mut ok = None;
        for f in [0.0, 0.5, 0.75, 0.95] {
            let n_erase = (plan.nsym as f64 * f) as usize;
            let mut erase = order[..n_erase].to_vec();
            erase.sort_unstable();
            if let Ok((msg, errata)) = rs::decode(block, plan.nsym, &erase) {
                ok = Some((msg, n_erase + 2 * errata.saturating_sub(n_erase)));
                break;
            }
        }
        let (msg, cost) = ok?;
        out.extend(msg);
        costs.push(cost);
    }
    Some((out, costs))
}

// ---------- Charge utile ----------

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

pub fn pack_payload(body: &[u8], codec: u8) -> Vec<u8> {
    let mut p = vec![FORMAT_VERSION, codec];
    p.extend((body.len() as u16).to_be_bytes());
    p.extend(body);
    let crc = crc32(&p);
    p.extend(crc.to_be_bytes());
    p
}

pub fn unpack_payload(data: &[u8]) -> Option<(u8, Vec<u8>)> {
    if data.len() < 8 || data[0] != FORMAT_VERSION {
        return None;
    }
    let len = u16::from_be_bytes([data[2], data[3]]) as usize;
    if data.len() < 8 + len {
        return None;
    }
    let crc = u32::from_be_bytes(data[4 + len..8 + len].try_into().ok()?);
    (crc32(&data[..4 + len]) == crc).then(|| (data[1], data[4..4 + len].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn format_aller_retour() {
        for n in [41, 256, 1023] {
            for lvl in 0..4 {
                let mut f = encode_format(n, lvl);
                f[3] ^= 0xff; // une erreur
                assert_eq!(decode_format(&f), Some((n, lvl)));
            }
        }
    }
    #[test]
    fn crc_standard() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
