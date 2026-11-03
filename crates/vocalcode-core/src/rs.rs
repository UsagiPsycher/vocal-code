//! Reed-Solomon sur GF(256) : polynôme primitif 0x11d, générateur 2, fcr = 0,
//! code systématique (parité ajoutée après le message). Décodage erreurs + effacements.

const PRIM: u16 = 0x11d;

struct Gf {
    exp: [u8; 512],
    log: [u8; 256],
}

fn gf() -> &'static Gf {
    use std::sync::OnceLock;
    static GF: OnceLock<Gf> = OnceLock::new();
    GF.get_or_init(|| {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u16 = 1;
        for i in 0..255 {
            exp[i] = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= PRIM;
            }
        }
        for i in 255..512 {
            exp[i] = exp[i - 255];
        }
        Gf { exp, log }
    })
}

#[inline]
fn mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let g = gf();
    g.exp[g.log[a as usize] as usize + g.log[b as usize] as usize]
}

fn div(a: u8, b: u8) -> u8 {
    assert!(b != 0);
    if a == 0 {
        return 0;
    }
    let g = gf();
    g.exp[(g.log[a as usize] as usize + 255 - g.log[b as usize] as usize) % 255]
}

fn pow(x: u8, p: i32) -> u8 {
    let g = gf();
    g.exp[((g.log[x as usize] as i32 * p).rem_euclid(255)) as usize]
}

fn inv(x: u8) -> u8 {
    div(1, x)
}

fn poly_scale(p: &[u8], x: u8) -> Vec<u8> {
    p.iter().map(|&c| mul(c, x)).collect()
}

fn poly_add(p: &[u8], q: &[u8]) -> Vec<u8> {
    let n = p.len().max(q.len());
    let mut r = vec![0u8; n];
    for (i, &c) in p.iter().enumerate() {
        r[i + n - p.len()] = c;
    }
    for (i, &c) in q.iter().enumerate() {
        r[i + n - q.len()] ^= c;
    }
    r
}

fn poly_mul(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut r = vec![0u8; p.len() + q.len() - 1];
    for (j, &qj) in q.iter().enumerate() {
        for (i, &pi) in p.iter().enumerate() {
            r[i + j] ^= mul(pi, qj);
        }
    }
    r
}

fn poly_eval(p: &[u8], x: u8) -> u8 {
    let mut y = p[0];
    for &c in &p[1..] {
        y = mul(y, x) ^ c;
    }
    y
}

fn generator(nsym: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 0..nsym {
        g = poly_mul(&g, &[1, pow(2, i as i32)]);
    }
    g
}

/// Ajoute `nsym` octets de parité à `msg`.
pub fn encode(msg: &[u8], nsym: usize) -> Vec<u8> {
    assert!(msg.len() + nsym <= 255);
    let gen = generator(nsym);
    let mut out = msg.to_vec();
    out.resize(msg.len() + nsym, 0);
    for i in 0..msg.len() {
        let coef = out[i];
        if coef != 0 {
            for j in 1..gen.len() {
                out[i + j] ^= mul(gen[j], coef);
            }
        }
    }
    out[..msg.len()].copy_from_slice(msg);
    out
}

#[derive(Debug)]
pub struct RsError;

fn syndromes(msg: &[u8], nsym: usize) -> Vec<u8> {
    let mut s = vec![0u8; nsym + 1];
    for i in 0..nsym {
        s[i + 1] = poly_eval(msg, pow(2, i as i32));
    }
    s
}

fn errata_locator(coef_pos: &[usize]) -> Vec<u8> {
    let mut loc = vec![1u8];
    for &p in coef_pos {
        loc = poly_mul(&loc, &poly_add(&[1], &[pow(2, p as i32), 0]));
    }
    loc
}

fn poly_div_rem(dividend: &[u8], divisor: &[u8]) -> Vec<u8> {
    let mut out = dividend.to_vec();
    if dividend.len() < divisor.len() {
        return out;
    }
    for i in 0..dividend.len() - (divisor.len() - 1) {
        let coef = out[i];
        if coef != 0 {
            for j in 1..divisor.len() {
                if divisor[j] != 0 {
                    out[i + j] ^= mul(divisor[j], coef);
                }
            }
        }
    }
    out[dividend.len() - (divisor.len() - 1)..].to_vec()
}

fn correct_errata(msg: &mut [u8], synd: &[u8], err_pos: &[usize]) -> Result<(), RsError> {
    let n = msg.len();
    let coef_pos: Vec<usize> = err_pos.iter().map(|&p| n - 1 - p).collect();
    let loc = errata_locator(&coef_pos);
    let mut synd_rev = synd.to_vec();
    synd_rev.reverse();
    let mut divisor = vec![1u8];
    divisor.resize(loc.len() + 1, 0); // x^(nerr + 1)
    let mut eval = poly_div_rem(&poly_mul(&synd_rev, &loc), &divisor);
    eval.reverse();

    let xs: Vec<u8> = coef_pos.iter().map(|&c| pow(2, -(255 - c as i32))).collect();
    for (i, &xi) in xs.iter().enumerate() {
        let xi_inv = inv(xi);
        let mut prime = 1u8;
        for (j, &xj) in xs.iter().enumerate() {
            if j != i {
                prime = mul(prime, 1 ^ mul(xi_inv, xj));
            }
        }
        if prime == 0 {
            return Err(RsError);
        }
        let mut ev = eval.clone();
        ev.reverse();
        let y = mul(pow(xi, 1), poly_eval(&ev, xi_inv));
        msg[err_pos[i]] ^= div(y, prime);
    }
    Ok(())
}

fn forney_syndromes(synd: &[u8], pos: &[usize], n: usize) -> Vec<u8> {
    let mut f = synd[1..].to_vec();
    for &p in pos {
        let x = pow(2, (n - 1 - p) as i32);
        for j in 0..f.len().saturating_sub(1) {
            f[j] = mul(f[j], x) ^ f[j + 1];
        }
    }
    f
}

fn error_locator(synd: &[u8], nsym: usize, erase_count: usize) -> Result<Vec<u8>, RsError> {
    let mut err_loc = vec![1u8];
    let mut old_loc = vec![1u8];
    let shift = synd.len().saturating_sub(nsym);
    for i in 0..nsym - erase_count {
        let k = i + shift;
        let mut delta = synd[k];
        for j in 1..err_loc.len() {
            delta ^= mul(err_loc[err_loc.len() - 1 - j], synd[k - j]);
        }
        old_loc.push(0);
        if delta != 0 {
            if old_loc.len() > err_loc.len() {
                let new_loc = poly_scale(&old_loc, delta);
                old_loc = poly_scale(&err_loc, inv(delta));
                err_loc = new_loc;
            }
            err_loc = poly_add(&err_loc, &poly_scale(&old_loc, delta));
        }
    }
    while err_loc.len() > 1 && err_loc[0] == 0 {
        err_loc.remove(0);
    }
    let errs = err_loc.len() - 1;
    if errs * 2 + erase_count > nsym {
        return Err(RsError);
    }
    Ok(err_loc)
}

fn find_errors(err_loc_rev: &[u8], n: usize) -> Result<Vec<usize>, RsError> {
    let errs = err_loc_rev.len() - 1;
    let mut pos = Vec::new();
    for i in 0..n {
        if poly_eval(err_loc_rev, pow(2, i as i32)) == 0 {
            pos.push(n - 1 - i);
        }
    }
    if pos.len() != errs {
        return Err(RsError);
    }
    Ok(pos)
}

/// Décode un bloc ; `erase` = positions connues comme douteuses.
/// Renvoie (message, nombre total de symboles corrigés ou effacés).
pub fn decode(block: &[u8], nsym: usize, erase: &[usize]) -> Result<(Vec<u8>, usize), RsError> {
    if erase.len() > nsym || block.len() <= nsym {
        return Err(RsError);
    }
    let mut msg = block.to_vec();
    for &p in erase {
        msg[p] = 0;
    }
    let synd = syndromes(&msg, nsym);
    if synd.iter().all(|&s| s == 0) {
        return Ok((msg[..msg.len() - nsym].to_vec(), erase.len()));
    }
    let fsynd = forney_syndromes(&synd, erase, msg.len());
    let loc = error_locator(&fsynd, nsym, erase.len())?;
    let mut rev = loc.clone();
    rev.reverse();
    let err_pos = find_errors(&rev, msg.len())?;
    let mut all: Vec<usize> = erase.to_vec();
    all.extend(&err_pos);
    correct_errata(&mut msg, &synd, &all)?;
    if syndromes(&msg, nsym).iter().any(|&s| s != 0) {
        return Err(RsError);
    }
    Ok((msg[..msg.len() - nsym].to_vec(), all.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_reedsolo() {
        let msg: Vec<u8> = (0..20).collect();
        let cw = encode(&msg, 10);
        assert_eq!(&cw[20..], &[41, 128, 63, 252, 188, 143, 16, 185, 191, 79]);
    }

    #[test]
    fn erreurs_et_effacements() {
        let msg: Vec<u8> = (0..150).map(|i| (i * 7 + 3) as u8).collect();
        let nsym = 100;
        let cw = encode(&msg, nsym);
        // 30 erreurs + 40 effacements = 2*30 + 40 = 100 <= nsym
        let mut bad = cw.clone();
        for i in 0..30 {
            bad[i * 8 + 1] ^= 0x5a;
        }
        let erase: Vec<usize> = (0..40).map(|i| i * 6 + 3).filter(|p| (p % 8) != 1).take(40).collect();
        for &p in &erase {
            bad[p] = 0xff;
        }
        let (out, _) = decode(&bad, nsym, &erase).expect("décodable");
        assert_eq!(out, msg);
        // 51 erreurs : au-delà de la capacité
        let mut worse = cw.clone();
        for i in 0..51 {
            worse[i * 4] ^= 0x33;
        }
        assert!(decode(&worse, nsym, &[]).map(|(m, _)| m != msg).unwrap_or(true));
    }
}
