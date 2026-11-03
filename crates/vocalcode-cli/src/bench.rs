//! Banc de robustesse : abîme l'image du code de façon contrôlée puis tente de la décoder.

use image::GrayImage;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;
use vocalcode_core::geom::fit_homography;
use vocalcode_core::imgops::{gaussian, FImage};
use vocalcode_core::{build_grid, decode_image, grid_to_image};

pub const HI: u32 = 8; // px/module du rendu de travail, avant réduction
const PAPER: f32 = 230.0;

#[derive(Clone, Copy)]
pub struct Params {
    pub ppm: f64,
    pub blur: f32,
    pub persp: f64,
    pub arc: f64,
    pub light: f32,
    pub glare: f32,
    pub stain: f64,
    pub noise: f32,
    pub jpeg: u8,
}

impl Default for Params {
    fn default() -> Self {
        Self { ppm: 4.0, blur: 0.0, persp: 0.0, arc: 0.0, light: 0.0, glare: 0.0, stain: 0.0, noise: 2.0, jpeg: 90 }
    }
}

fn remap(src: &FImage, w: usize, h: usize, border: f32, f: impl Fn(f64, f64) -> Option<(f64, f64)> + Sync) -> FImage {
    let data: Vec<f32> = (0..w * h).into_par_iter().map(|i| {
        let (x, y) = ((i % w) as f64, (i / w) as f64);
        match f(x, y) {
            Some((sx, sy)) => src.sample_or(sx, sy, border),
            None => border,
        }
    }).collect();
    FImage { w, h, data }
}

fn pad(img: &FImage, p: usize, v: f32) -> FImage {
    let (w, h) = (img.w + 2 * p, img.h + 2 * p);
    let mut out = FImage::new(w, h, v);
    for y in 0..img.h {
        out.data[(y + p) * w + p..(y + p) * w + p + img.w].copy_from_slice(&img.data[y * img.w..(y + 1) * img.w]);
    }
    out
}

/// Code collé sur un cylindre (poteau) : le code couvre `arc_deg` degrés de l'arc.
fn cylinder(img: &FImage, arc_deg: f64) -> FImage {
    if arc_deg <= 0.0 {
        return img.clone();
    }
    let w = img.w as f64;
    let phi = arc_deg.to_radians();
    let r = w / phi;
    let out_half = r * (phi / 2.0).min(std::f64::consts::FRAC_PI_2 - 1e-3).sin();
    let col: Vec<(f64, f32)> = (0..img.w).map(|u| {
        let xo = -out_half + 2.0 * out_half * u as f64 / (w - 1.0);
        let a = (xo / r).clamp(-1.0, 1.0).asin();
        (r * a + w / 2.0, (0.55 + 0.45 * a.cos()) as f32)
    }).collect();
    let mut out = remap(img, img.w, img.h, PAPER, |x, y| Some((col[x as usize].0, y)));
    for y in 0..out.h {
        for x in 0..out.w {
            out.data[y * out.w + x] *= col[x].1;
        }
    }
    out
}

fn perspective(img: &FImage, amount: f64, rng: &mut StdRng) -> FImage {
    if amount <= 0.0 {
        return img.clone();
    }
    let p = (amount * img.w.max(img.h) as f64) as usize;
    let big = pad(img, p, PAPER);
    let (w, h, pf) = (img.w as f64, img.h as f64, p as f64);
    let src = [[pf, pf], [pf + w, pf], [pf + w, pf + h], [pf, pf + h]];
    let dst: Vec<[f64; 2]> = src.iter().map(|s| {
        [s[0] + rng.gen_range(-amount..amount) * w, s[1] + rng.gen_range(-amount..amount) * h]
    }).collect();
    let inv = fit_homography(&dst, &src).unwrap();
    remap(&big, big.w, big.h, PAPER, |x, y| { let s = inv.apply([x, y]); Some((s[0], s[1])) })
}

fn lighting(img: &mut FImage, strength: f32, glare: f32, rng: &mut StdRng) {
    let (w, h) = (img.w as f32, img.h as f32);
    let ang: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    let (gx, gy): (f32, f32) = (rng.gen_range(0.3..0.7), rng.gen_range(0.3..0.7));
    let (c, s) = (ang.cos(), ang.sin());
    let corners = [0.0, c, s, c + s];
    let (gmin, gmax) = (corners.iter().cloned().fold(f32::MAX, f32::min), corners.iter().cloned().fold(f32::MIN, f32::max));
    let r = glare * w.min(h);
    for y in 0..img.h {
        for x in 0..img.w {
            let v = &mut img.data[y * img.w + x];
            let (xf, yf) = (x as f32, y as f32);
            if strength > 0.0 {
                let g = (c * xf / w + s * yf / h - gmin) / (gmax - gmin + 1e-6);
                *v *= 1.0 - strength * g;
            }
            if glare > 0.0 {
                let dx = (xf - gx * w) / r;
                let dy = (yf - gy * h) / (0.6 * r);
                *v += (-(dx * dx + dy * dy)).exp() * 255.0 * 0.9;
            }
        }
    }
}

fn stains(img: &mut FImage, frac: f64, rng: &mut StdRng, bx: (usize, usize, usize, usize)) {
    if frac <= 0.0 {
        return;
    }
    let (x0, y0, x1, y1) = bx;
    let area = ((x1 - x0) * (y1 - y0)) as f64;
    let mut mask = vec![false; img.w * img.h];
    let mut covered = 0usize;
    while (covered as f64) < frac * area {
        let a = rng.gen_range(0.02..0.06) * (x1 - x0) as f64;
        let b = a * rng.gen_range(0.4..1.0);
        let (cx, cy) = (rng.gen_range(x0 as f64..x1 as f64), rng.gen_range(y0 as f64..y1 as f64));
        let t: f64 = rng.gen_range(0.0..std::f64::consts::PI);
        let (ct, st) = (t.cos(), t.sin());
        for y in ((cy - a) as isize).max(0)..=((cy + a) as isize).min(img.h as isize - 1) {
            for x in ((cx - a) as isize).max(0)..=((cx + a) as isize).min(img.w as isize - 1) {
                let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                let (u, v) = ((dx * ct + dy * st) / a, (-dx * st + dy * ct) / b);
                if u * u + v * v <= 1.0 {
                    let i = y as usize * img.w + x as usize;
                    if !mask[i] {
                        mask[i] = true;
                        if (x0..x1).contains(&(x as usize)) && (y0..y1).contains(&(y as usize)) {
                            covered += 1;
                        }
                    }
                }
            }
        }
    }
    let tone = FImage { w: img.w, h: img.h,
                        data: (0..img.w * img.h).map(|_| if rng.gen_bool(0.5) { 40.0 } else { 220.0 }).collect() };
    let tone = gaussian(&tone, 6.0);
    for (i, m) in mask.iter().enumerate() {
        if *m {
            img.data[i] = tone.data[i];
        }
    }
}

/// Pour une image qui contient déjà sa marge blanche (PNG généré).
pub fn degrade_framed(base: &GrayImage, p: &Params, seed: u64) -> GrayImage {
    degrade(base, p, seed)
}

pub fn degrade(base: &GrayImage, p: &Params, seed: u64) -> GrayImage {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut img = FImage::from_gray(base);
    img.data.iter_mut().for_each(|v| *v = *v * (175.0 / 255.0) + 45.0); // encre et papier imparfaits
    let margin = HI as usize * 6;
    let (w, h) = (img.w, img.h);
    let mut img = pad(&img, margin, PAPER);
    stains(&mut img, p.stain, &mut rng, (margin, margin, margin + w, margin + h));
    let img = cylinder(&img, p.arc);
    let mut img = perspective(&img, p.persp, &mut rng);
    lighting(&mut img, p.light, p.glare, &mut rng);
    let img = gaussian(&img, p.blur * HI as f32);
    // réduction à la résolution « caméra » (préfiltre ~ moyenne de zone)
    let s = p.ppm / HI as f64;
    let pre = gaussian(&img, (0.29 / s) as f32);
    let (ow, oh) = ((img.w as f64 * s) as usize, (img.h as f64 * s) as usize);
    let mut small = remap(&pre, ow, oh, PAPER, |x, y| Some(((x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5)));
    let normal = Normal::new(0.0f32, p.noise.max(1e-6)).unwrap();
    small.data.iter_mut().for_each(|v| *v += normal.sample(&mut rng));
    let gray = small.to_gray();
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, p.jpeg).encode_image(&gray).unwrap();
    image::load_from_memory(&buf).unwrap().to_luma8()
}

pub fn scenarios() -> Vec<(&'static str, Params)> {
    let d = Params::default();
    vec![
        ("propre, 4 px/module", d),
        ("3 px/module", Params { ppm: 3.0, ..d }),
        ("2,5 px/module", Params { ppm: 2.5, ..d }),
        ("flou 0,4 module", Params { blur: 0.4, ..d }),
        ("flou 0,6 module", Params { blur: 0.6, ..d }),
        ("perspective 15 %", Params { persp: 0.15, ..d }),
        ("perspective 25 %", Params { persp: 0.25, ..d }),
        ("poteau, arc 60°", Params { arc: 60.0, ..d }),
        ("poteau, arc 100°", Params { arc: 100.0, ..d }),
        ("éclairage inégal + reflet", Params { light: 0.6, glare: 0.12, ..d }),
        ("taches 5 %", Params { stain: 0.05, ..d }),
        ("taches 10 %", Params { stain: 0.10, ..d }),
        ("bruit fort + JPEG 50", Params { noise: 10.0, jpeg: 50, ..d }),
        ("réaliste combiné", Params { ppm: 4.0, blur: 0.3, persp: 0.1, arc: 45.0, light: 0.4, glare: 0.08,
                                      stain: 0.02, noise: 5.0, jpeg: 75 }),
        ("dur combiné, 4 px/module", Params { ppm: 4.0, blur: 0.4, persp: 0.18, arc: 70.0, light: 0.6, glare: 0.12,
                                              stain: 0.05, noise: 8.0, jpeg: 60 }),
        ("dur combiné", Params { ppm: 3.0, blur: 0.4, persp: 0.18, arc: 70.0, light: 0.6, glare: 0.12,
                                 stain: 0.05, noise: 8.0, jpeg: 60 }),
    ]
}

pub fn run(body: &[u8], level: u8, trials: usize, examples: Option<&Path>, only: Option<&str>) -> anyhow::Result<()> {
    let grid = build_grid(body, level).map_err(anyhow::Error::msg)?;
    let base = grid_to_image(&grid, HI);
    println!("grille {0}x{0}, ecc {level}, {1} octets\n", grid.n, body.len());
    println!("{:<28} {:>9} {:>24} {:>8}", "scénario", "réussite", "correction (méd./max)", "temps");
    if let Some(ex) = examples {
        std::fs::create_dir_all(ex)?;
    }
    for (i, (name, p)) in scenarios().into_iter().enumerate() {
        if only.is_some_and(|o| !name.contains(o)) {
            continue;
        }
        let results: Vec<(Result<f64, String>, f64)> = (0..trials).into_par_iter().map(|k| {
            let img = degrade(&base, &p, 1000 * i as u64 + k as u64);
            if k == 0 {
                if let Some(ex) = examples {
                    let _ = img.save(ex.join(format!("{i:02}.jpg")));
                }
            }
            let t0 = Instant::now();
            let r = match decode_image(&img) {
                Ok(r) if r.body == body => Ok(r.worst_usage()),
                Ok(_) => Err("contenu différent".to_string()),
                Err(e) => {
                    if std::env::var_os("VC_VERBOSE").is_some() {
                        eprintln!("  [{name} #{k}] {e}");
                        if let Some(ex) = examples {
                            let _ = img.save(ex.join(format!("echec_{i:02}_{k}.png")));
                        }
                    }
                    Err(e.0.split(" ;").next().unwrap_or("").to_string())
                }
            };
            (r, t0.elapsed().as_secs_f64())
        }).collect();
        let mut used: Vec<f64> = results.iter().filter_map(|r| r.0.as_ref().ok().copied()).collect();
        used.sort_by(f64::total_cmp);
        let mut times: Vec<f64> = results.iter().map(|r| r.1).collect();
        times.sort_by(f64::total_cmp);
        let marge = if used.is_empty() { "-".into() } else {
            format!("{:.0}% / {:.0}%", used[used.len() / 2] * 100.0, used[used.len() - 1] * 100.0)
        };
        let first_err = results.iter().find_map(|r| r.0.as_ref().err());
        println!("{:<28} {:>4}/{:<4} {:>24} {:>6.0}ms{}", name, used.len(), trials, marge, times[times.len() / 2] * 1000.0,
                 first_err.map(|e| format!("   [{e}]")).unwrap_or_default());
    }
    Ok(())
}
