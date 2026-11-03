//! Image (photo) -> charge utile.
//!
//! 1. repérer les repères de coin (carrés concentriques) ; 3 sur 4 suffisent ;
//! 2. lire le bloc de format : il donne la taille de grille, le niveau de correction
//!    et tranche l'orientation ;
//! 3. recaler chaque repère d'alignement, de proche en proche depuis les coins
//!    (perspective, courbure) ;
//! 4. position de chaque module = homographie globale + spline de plaque mince ;
//! 5. échantillonner, égaliser, seuiller localement, démêler, corriger, vérifier le CRC.

use crate::ecc::{block_plan, decode_format, rs_decode, unpack_payload};
use crate::geom::{fit_homography, fit_homography_weighted, fit_map, Homography, Pt, Tps};
use crate::imgops::{box_mean, masked_mean, FImage};
use crate::layout::{layout, marker_pattern, slot_permutation, whitening, Kind, Layout, FORMAT_SIDE,
                    MARKER, MAX_N, MIN_N};
use image::GrayImage;
use imageproc::contours::{find_contours, BorderType};
use imageproc::geometry::{approximate_polygon_dp, arc_length, convex_hull};
use imageproc::point::Point;

#[derive(Debug)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DecodeError {}

fn err<T>(m: impl Into<String>) -> Result<T, DecodeError> {
    Err(DecodeError(m.into()))
}

#[derive(Clone, Debug)]
pub struct Finder {
    pub center: Pt,
    pub corners: Option<[Pt; 4]>,
    pub module: f64,
}

#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub codec: u8,
    pub body: Vec<u8>,
    pub n: usize,
    pub level: u8,
    pub corners_found: usize,
    pub markers_found: usize,
    pub markers_total: usize,
    /// coût de correction par bloc, en symboles de parité
    pub block_costs: Vec<usize>,
    pub nsym: usize,
}

impl DecodeResult {
    /// Part de la capacité de correction consommée par le pire bloc.
    pub fn worst_usage(&self) -> f64 {
        *self.block_costs.iter().max().unwrap_or(&0) as f64 / self.nsym as f64
    }
}

// ---------- 1. Repères de coin ----------

fn poly_area(p: &[Point<i32>]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0i64;
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        s += a.x as i64 * b.y as i64 - b.x as i64 * a.y as i64;
    }
    (s as f64).abs() / 2.0
}

fn centroid(p: &[Point<i32>]) -> Pt {
    let n = p.len();
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (p0, p1) = (p[i], p[(i + 1) % n]);
        let cr = (p0.x as f64) * (p1.y as f64) - (p1.x as f64) * (p0.y as f64);
        a += cr;
        cx += (p0.x + p1.x) as f64 * cr;
        cy += (p0.y + p1.y) as f64 * cr;
    }
    if a.abs() < 1e-9 {
        let k = n as f64;
        return [p.iter().map(|q| q.x as f64).sum::<f64>() / k, p.iter().map(|q| q.y as f64).sum::<f64>() / k];
    }
    [cx / (3.0 * a), cy / (3.0 * a)]
}

fn candidates_in(gray: &GrayImage, out: &mut Vec<Finder>) {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let g = FImage::from_gray(gray);
    let blur = crate::imgops::gaussian(&g, 3.0);
    let sharp = FImage { w, h, data: g.data.iter().zip(&blur.data).map(|(a, b)| 2.5 * a - 1.5 * b).collect() }.to_gray();
    for src in [gray, &sharp] {
        for frac in [0.02, 0.05, 0.12] {
            let bs = ((w.min(h) as f64 * frac) as u32 | 1).max(11);
            let th = imageproc::contrast::adaptive_threshold(src, bs / 2, 5);
            let binv = GrayImage::from_fn(th.width(), th.height(), |x, y| image::Luma([255 - th.get_pixel(x, y)[0]]));
            let cs = find_contours::<i32>(&binv);
            let mut children = vec![Vec::new(); cs.len()];
            for (i, c) in cs.iter().enumerate() {
                if let Some(p) = c.parent {
                    children[p].push(i);
                }
            }
            // cœur du repère : un « trou » blanc (anneau 5x5) contenant un seul carré noir 3x3.
            // On ne s'appuie pas sur le contour extérieur, qui fusionne avec les données si le flou
            // efface l'anneau séparateur.
            for (h, c) in cs.iter().enumerate() {
                if c.border_type != BorderType::Hole || children[h].len() != 1 || c.points.len() < 8 {
                    continue;
                }
                let g = children[h][0];
                if !children[g].is_empty() {
                    continue;
                }
                let a1 = poly_area(&c.points);
                let a2 = poly_area(&cs[g].points);
                if a1 < 60.0 || a2 <= 0.0 {
                    continue;
                }
                // théorie 25:9 ; aux petites tailles le bord des pixels gonfle le rapport
                let r = a1 / a2;
                if !(1.8 < r && r < 5.5) {
                    continue;
                }
                let hull = convex_hull(c.points.clone());
                if a1 / poly_area(&hull).max(1.0) < 0.85 {
                    continue;
                }
                let ghull = convex_hull(cs[g].points.clone());
                if a2 / poly_area(&ghull).max(1.0) < 0.75 {
                    continue;
                }
                // le carré central doit être à peu près au centre du trou
                let (ch, cg) = (centroid(&c.points), centroid(&cs[g].points));
                if dist(ch, cg) > 0.15 * a1.sqrt() {
                    continue;
                }
                let eps = 0.08 * arc_length(&hull, true);
                let approx = approximate_polygon_dp(&hull, eps, true);
                let corners = (approx.len() == 4).then(|| {
                    let mut k = [[0.0; 2]; 4];
                    for (j, p) in approx.iter().enumerate() {
                        k[j] = [p.x as f64, p.y as f64];
                    }
                    k
                });
                out.push(Finder { center: cg, corners, module: (a1.sqrt() + 1.0) / 5.0 });
            }
        }
    }
}

pub fn finder_candidates(gray: &GrayImage) -> Vec<Finder> {
    // détection sur une version réduite si l'image est grande
    let max_dim = gray.width().max(gray.height());
    let limit = 1800;
    let mut found = Vec::new();
    if max_dim > limit {
        let s = limit as f64 / max_dim as f64;
        let small = image::imageops::resize(gray, (gray.width() as f64 * s) as u32, (gray.height() as f64 * s) as u32,
                                            image::imageops::FilterType::Triangle);
        candidates_in(&small, &mut found);
        let k = 1.0 / s;
        for f in &mut found {
            f.center = [(f.center[0] + 0.5) * k - 0.5, (f.center[1] + 0.5) * k - 0.5];
            f.module *= k;
            if let Some(c) = &mut f.corners {
                for p in c.iter_mut() {
                    *p = [(p[0] + 0.5) * k - 0.5, (p[1] + 0.5) * k - 0.5];
                }
            }
        }
    } else {
        candidates_in(gray, &mut found);
    }
    // dédoublonnage
    found.sort_by(|a, b| b.module.total_cmp(&a.module));
    let mut uniq: Vec<Finder> = Vec::new();
    for f in found {
        if uniq.iter().all(|u| dist(f.center, u.center) > 2.0 * u.module) {
            uniq.push(f);
        }
    }
    uniq
}

fn dist(a: Pt, b: Pt) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Meilleur triplet formant un angle droit : (précédent, sommet, suivant) dans le sens horaire.
fn pick_triple(c: &[Finder]) -> Option<[Finder; 3]> {
    let c = &c[..c.len().min(12)];
    let mut best: Option<([Finder; 3], f64)> = None;
    for i in 0..c.len() {
        for j in i + 1..c.len() {
            for k in j + 1..c.len() {
                let tri = [&c[i], &c[j], &c[k]];
                let mods: Vec<f64> = tri.iter().map(|f| f.module).collect();
                let (mmin, mmax) = (mods.iter().cloned().fold(f64::MAX, f64::min), mods.iter().cloned().fold(0.0, f64::max));
                if mmax / mmin > 3.0 {
                    continue;
                }
                let mmean = mods.iter().sum::<f64>() / 3.0;
                for v in 0..3 {
                    let (a, b) = (tri[(v + 1) % 3], tri[(v + 2) % 3]);
                    let va = [a.center[0] - tri[v].center[0], a.center[1] - tri[v].center[1]];
                    let vb = [b.center[0] - tri[v].center[0], b.center[1] - tri[v].center[1]];
                    let (la, lb) = ((va[0].powi(2) + va[1].powi(2)).sqrt(), (vb[0].powi(2) + vb[1].powi(2)).sqrt());
                    if la.min(lb) < 10.0 * mmean {
                        continue;
                    }
                    let cos = (va[0] * vb[0] + va[1] * vb[1]).abs() / (la * lb);
                    let ratio = la.max(lb) / la.min(lb);
                    if cos > 0.7 || ratio > 3.0 {
                        continue;
                    }
                    let score = cos + (ratio - 1.0) * 0.5 + (mmax / mmin - 1.0) * 0.5;
                    if best.as_ref().map_or(true, |b| score < b.1) {
                        let cross = va[0] * vb[1] - va[1] * vb[0];
                        let (nxt, prev) = if cross > 0.0 { (a, b) } else { (b, a) };
                        best = Some(([prev.clone(), tri[v].clone(), nxt.clone()], score));
                    }
                }
            }
        }
    }
    best.map(|b| b.0)
}

// ---------- ancres (points connus module <-> image) ----------

#[derive(Default, Clone)]
struct Anchors {
    module: Vec<Pt>,
    image: Vec<Pt>,
}

impl Anchors {
    fn push(&mut self, m: Pt, i: Pt) {
        self.module.push(m);
        self.image.push(i);
    }
}

/// Hypothèse d'orientation : le sommet de l'angle droit est le coin n° `rot` (sens horaire depuis TL).
fn assign(tri: &[Finder; 3], n: usize, rot: usize) -> Vec<(Finder, Pt)> {
    let cc = Layout::corner_centers(n);
    let c = |i: usize| -> Pt { let p = cc[i % 4]; [p.0, p.1] };
    vec![(tri[0].clone(), c(rot + 3)), (tri[1].clone(), c(rot)), (tri[2].clone(), c(rot + 1))]
}

fn finder_anchors(pairs: &[(Finder, Pt)]) -> Anchors {
    let mut a = Anchors::default();
    for (f, m) in pairs {
        a.push(*m, f.center);
    }
    let Some(h) = fit_map(&a.module.clone(), &a.image.clone()) else { return a };
    for (f, m) in pairs {
        let Some(corners) = f.corners else { continue };
        // coins de l'anneau blanc intérieur (5x5 modules)
        let ideal = [[m[0] - 2.5, m[1] - 2.5], [m[0] + 2.5, m[1] - 2.5], [m[0] + 2.5, m[1] + 2.5], [m[0] - 2.5, m[1] + 2.5]];
        for id in ideal {
            let p = h.apply(id);
            let (j, d) = corners.iter().enumerate().map(|(j, c)| (j, dist(*c, p)))
                .min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            if d < 1.5 * f.module {
                a.push(id, corners[j]);
            }
        }
    }
    a
}

// ---------- 2. Bloc de format ----------

fn read_format(contrast: &FImage, tri: &[Finder; 3], n_est: f64) -> Option<(usize, u8, Vec<(Finder, Pt)>)> {
    let s = FORMAT_SIDE;
    let lo = ((n_est * 0.6) as usize).max(MIN_N);
    let hi = ((n_est * 1.4) as usize + 1).min(MAX_N + 1);
    let mut sizes: Vec<usize> = (lo..hi).collect();
    sizes.sort_by(|a, b| (*a as f64 - n_est).abs().total_cmp(&(*b as f64 - n_est).abs()));
    for n in sizes {
        let lay = layout(n);
        for rot in 0..4 {
            let pairs = assign(tri, n, rot);
            let a = finder_anchors(&pairs);
            let mut maps = vec![fit_map(&a.module[..3], &a.image[..3])];
            if a.module.len() >= 6 {
                maps.push(fit_homography(&a.module, &a.image));
            }
            for h in maps.into_iter().flatten() {
                let per_copy: Vec<Vec<f32>> = lay.format_areas.iter().map(|&(ax, ay)| {
                    (0..s * s).map(|i| {
                        let (mx, my) = ((ax + i % s) as f64 + 0.5, (ay + i / s) as f64 + 0.5);
                        let f = |dx: f64, dy: f64| { let p = h.apply([mx + dx, my + dy]); contrast.sample(p[0], p[1]) };
                        // même égaliseur que pour les données
                        f(0.0, 0.0) - 0.2 * (f(-1.0, 0.0) + f(1.0, 0.0) + f(0.0, -1.0) + f(0.0, 1.0))
                    }).collect()
                }).collect();
                let sum: Vec<f32> = (0..s * s).map(|i| per_copy.iter().map(|c| c[i]).sum()).collect();
                for soft in std::iter::once(&sum).chain(per_copy.iter()) {
                    let mut cw = [0u8; 8];
                    for (i, v) in soft.iter().enumerate() {
                        if *v < 0.0 {
                            cw[i / 8] |= 1 << (7 - i % 8);
                        }
                    }
                    if let Some((nn, level)) = decode_format(&cw) {
                        if (MIN_N..=MAX_N).contains(&nn) && (nn as f64 - n_est).abs() < 0.45 * n_est {
                            let pairs = if nn != n { assign(tri, nn, rot) } else { pairs };
                            return Some((nn, level, pairs));
                        }
                    }
                }
            }
        }
    }
    None
}

// ---------- 3. Recalage des repères d'alignement ----------

const P: usize = 6; // pixels par module dans les vignettes redressées
const HALF: f64 = 6.5; // demi-côté de la vignette, en modules

struct Template {
    side: usize,
    centered: Vec<f32>,
    norm: f32,
}

fn template() -> &'static Template {
    use std::sync::OnceLock;
    static T: OnceLock<Template> = OnceLock::new();
    T.get_or_init(|| {
        let side = MARKER * P;
        let raw: Vec<f32> = (0..side * side)
            .map(|i| if marker_pattern((i % side) / P, (i / side) / P) == 1 { 0.0 } else { 255.0 })
            .collect();
        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        let centered: Vec<f32> = raw.iter().map(|v| v - mean).collect();
        let norm = centered.iter().map(|v| v * v).sum::<f32>().sqrt();
        Template { side, centered, norm }
    })
}

/// Cherche le repère autour de la position prédite ;
/// renvoie (score NCC, position dans le repère prédit en modules, position image).
fn refine_marker(gray: &FImage, h: &Homography, cx: usize, cy: usize) -> (f32, Pt, Pt) {
    let t = template();
    let size = (2.0 * HALF * P as f64) as usize;
    let (ox, oy) = (cx as f64 + 0.5 - HALF, cy as f64 + 0.5 - HALF);
    let mut patch = vec![0f32; size * size];
    for v in 0..size {
        for u in 0..size {
            let p = h.apply([ox + (u as f64 + 0.5) / P as f64, oy + (v as f64 + 0.5) / P as f64]);
            patch[v * size + u] = gray.sample(p[0], p[1]);
        }
    }
    // intégrales pour la moyenne et la variance locales
    let w1 = size + 1;
    let mut s1 = vec![0f64; w1 * w1];
    let mut s2 = vec![0f64; w1 * w1];
    for y in 0..size {
        let (mut r1, mut r2) = (0f64, 0f64);
        for x in 0..size {
            let v = patch[y * size + x] as f64;
            r1 += v;
            r2 += v * v;
            s1[(y + 1) * w1 + x + 1] = s1[y * w1 + x + 1] + r1;
            s2[(y + 1) * w1 + x + 1] = s2[y * w1 + x + 1] + r2;
        }
    }
    let ts = t.side;
    let rn = size - ts + 1;
    let npx = (ts * ts) as f64;
    let mut res = vec![0f32; rn * rn];
    for v in 0..rn {
        for u in 0..rn {
            let mut acc = 0f32;
            for j in 0..ts {
                let prow = &patch[(v + j) * size + u..(v + j) * size + u + ts];
                let trow = &t.centered[j * ts..(j + 1) * ts];
                acc += prow.iter().zip(trow).map(|(a, b)| a * b).sum::<f32>();
            }
            let box_sum = |s: &[f64]| s[(v + ts) * w1 + u + ts] - s[v * w1 + u + ts] - s[(v + ts) * w1 + u] + s[v * w1 + u];
            let (a, b) = (box_sum(&s1), box_sum(&s2));
            let var = (b - a * a / npx).max(1e-6);
            res[v * rn + u] = acc / (t.norm * var.sqrt() as f32);
        }
    }
    let (mut bu, mut bv, mut best) = (0, 0, f32::MIN);
    for (i, &r) in res.iter().enumerate() {
        if r > best {
            best = r;
            bu = i % rn;
            bv = i / rn;
        }
    }
    let quad = |l: f32, c: f32, r: f32| { let d = l - 2.0 * c + r; if d.abs() < 1e-9 { 0.0 } else { 0.5 * (l - r) / d } };
    let du = if bu > 0 && bu + 1 < rn { quad(res[bv * rn + bu - 1], best, res[bv * rn + bu + 1]) } else { 0.0 };
    let dv = if bv > 0 && bv + 1 < rn { quad(res[(bv - 1) * rn + bu], best, res[(bv + 1) * rn + bu]) } else { 0.0 };
    let half_t = ts as f64 / 2.0;
    let m = [ox + (bu as f64 + du as f64 + half_t) / P as f64, oy + (bv as f64 + dv as f64 + half_t) / P as f64];
    (best, m, h.apply(m))
}

/// Homographie locale autour de `target` : ancres proches, pondérées par la distance.
fn local_map(a: &Anchors, target: Pt, k: usize, exclude: Option<usize>) -> Option<Homography> {
    let mut idx: Vec<usize> = (0..a.module.len()).filter(|&i| Some(i) != exclude).collect();
    idx.sort_by(|&x, &y| dist(a.module[x], target).total_cmp(&dist(a.module[y], target)));
    idx.truncate(k);
    let m: Vec<Pt> = idx.iter().map(|&i| a.module[i]).collect();
    let im: Vec<Pt> = idx.iter().map(|&i| a.image[i]).collect();
    if m.len() >= 4 {
        let w: Vec<f64> = m.iter().map(|p| 1.0 / (1.0 + (dist(*p, target) / 25.0).powi(2))).collect();
        if let Some(h) = fit_homography_weighted(&m, &im, Some(&w)) {
            return Some(h);
        }
    }
    fit_map(&m, &im)
}

/// Écart (en modules) entre une ancre et la prédiction de ses voisines.
fn loo_error(a: &Anchors, i: usize) -> f64 {
    let Some(h) = local_map(a, a.module[i], 12, Some(i)) else { return f64::MAX };
    let m = a.module[i];
    let px = dist(h.apply(m), h.apply([m[0] + 1.0, m[1]])).max(1e-6);
    dist(h.apply(m), a.image[i]) / px
}

/// Retire les repères d'alignement incohérents avec leurs voisins (les ancres de coin sont gardées).
fn prune(a: &mut Anchors, n_fixed: usize, found: &mut Vec<(usize, usize)>) {
    for _ in 0..4 {
        let errs: Vec<f64> = (n_fixed..a.module.len()).map(|i| loo_error(a, i)).collect();
        let bad: Vec<usize> = errs.iter().enumerate().filter(|(_, e)| **e > 0.75).map(|(j, _)| j + n_fixed).collect();
        if bad.is_empty() {
            return;
        }
        // on retire d'abord les pires : un seul faux repère fausse aussi ses voisins
        let worst = errs.iter().cloned().fold(0.0, f64::max);
        let drop: Vec<usize> = bad.into_iter().filter(|&i| errs[i - n_fixed] > 0.5 * worst).collect();
        for &i in drop.iter().rev() {
            a.module.remove(i);
            a.image.remove(i);
            found.remove(i - n_fixed);
        }
    }
}

fn try_marker(gray: &FImage, a: &Anchors, c: (usize, usize), max_off: f64, trace: &mut Trace) -> Option<Pt> {
    let target = [c.0 as f64 + 0.5, c.1 as f64 + 0.5];
    let h = local_map(a, target, 12, None)?;
    let (score, m, pos) = refine_marker(gray, &h, c.0, c.1);
    trace.markers.push((h.apply(target), pos, score));
    let off = dist(m, target);
    // écart modéré : score correct suffit ; grand écart : il faut une corrélation forte
    ((score >= 0.45 && off <= max_off) || (score >= 0.65 && off <= 4.0)).then_some(pos)
}

/// Recale les repères d'alignement ; ajoute aussi les coins manquants retrouvés en cours de route.
fn locate_markers(gray: &FImage, lay: &Layout, anchors: &mut Anchors, corner_mods: &[Pt], cands: &[Finder],
                  mod_px: f64, trace: &mut Trace) -> (usize, usize) {
    let mut order = lay.marker_centers.clone();
    let d_to_corner = |c: &(usize, usize)| {
        corner_mods.iter().map(|m| dist(*m, [c.0 as f64 + 0.5, c.1 as f64 + 0.5])).fold(f64::MAX, f64::min)
    };
    order.sort_by(|a, b| d_to_corner(a).total_cmp(&d_to_corner(b)));

    // passe 1 : de proche en proche depuis les coins
    let mut n_fixed = anchors.module.len();
    let mut found: Vec<(usize, usize)> = Vec::new();
    for &c in &order {
        if let Some(pos) = try_marker(gray, anchors, c, 1.5, trace) {
            anchors.push([c.0 as f64 + 0.5, c.1 as f64 + 0.5], pos);
            found.push(c);
        }
    }
    prune(anchors, n_fixed, &mut found);

    // coins manquants : prédits localement, puis cherchés parmi les candidats
    let mut extra_corners = 0;
    for (cx, cy) in Layout::corner_centers(lay.n) {
        let c = [cx, cy];
        if corner_mods.contains(&c) {
            continue;
        }
        let Some(h) = local_map(anchors, c, 8, None) else { continue };
        let p = h.apply(c);
        let near = cands.iter()
            .filter(|f| f.module > 0.5 * mod_px && f.module < 2.0 * mod_px && dist(f.center, p) < 3.0 * mod_px)
            .min_by(|a, b| dist(a.center, p).total_cmp(&dist(b.center, p)));
        if let Some(f) = near {
            anchors.module.insert(n_fixed, c);
            anchors.image.insert(n_fixed, f.center);
            n_fixed += 1;
            extra_corners += 1;
            trace.corners.push(f.center);
        }
    }

    // passe 2 : repères manqués, désormais encadrés par des voisins sûrs
    for &c in &order {
        if found.contains(&c) {
            continue;
        }
        if let Some(pos) = try_marker(gray, anchors, c, 1.0, trace) {
            anchors.push([c.0 as f64 + 0.5, c.1 as f64 + 0.5], pos);
            found.push(c);
        }
    }
    prune(anchors, n_fixed, &mut found);
    (found.len(), extra_corners)
}

// ---------- 4-5. Échantillonnage et décodage ----------

fn module_positions(n: usize, a: &Anchors) -> Option<Vec<Pt>> {
    let h = fit_homography(&a.module, &a.image)?;
    let resid: Vec<Pt> = a.module.iter().zip(&a.image).map(|(m, i)| { let p = h.apply(*m); [i[0] - p[0], i[1] - p[1]] }).collect();
    let scale = dist(h.apply([0.0, 0.0]), h.apply([1.0, 0.0]));
    let tps = Tps::fit(&a.module, &resid, 0.5 * scale)?;
    let mut out = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let m = [x as f64 + 0.5, y as f64 + 0.5];
            let p = h.apply(m);
            let r = tps.eval(m);
            out.push([p[0] + r[0], p[1] + r[1]]);
        }
    }
    Some(out)
}

/// Informations de diagnostic sur la géométrie trouvée.
#[derive(Default)]
pub struct Trace {
    /// (position prédite, position trouvée, score) pour chaque repère d'alignement
    pub markers: Vec<(Pt, Pt, f32)>,
    pub corners: Vec<Pt>,
    pub modules: Vec<Pt>,
}

pub fn decode_image(gray: &GrayImage) -> Result<DecodeResult, DecodeError> {
    decode_image_traced(gray, &mut Trace::default())
}

pub fn decode_image_traced(gray: &GrayImage, trace: &mut Trace) -> Result<DecodeResult, DecodeError> {
    let cands = finder_candidates(gray);
    let Some(tri) = pick_triple(&cands) else {
        return err(format!("repères de coin introuvables ({} candidats)", cands.len()));
    };
    let mod_px = tri.iter().map(|f| f.module).sum::<f64>() / 3.0;
    let d = (dist(tri[2].center, tri[1].center) + dist(tri[0].center, tri[1].center)) / 2.0;
    let n_est = d / mod_px + 7.0;

    let g = FImage::from_gray(gray);
    let bg = box_mean(&g, ((mod_px * 9.0) as usize) | 1);
    let contrast = FImage { w: g.w, h: g.h, data: g.data.iter().zip(&bg.data).map(|(a, b)| a - b).collect() };

    let Some((n, level, pairs)) = read_format(&contrast, &tri, n_est) else {
        return err("bloc de format illisible");
    };
    let lay = layout(n);

    let mut anchors = finder_anchors(&pairs);
    let corner_mods: Vec<Pt> = pairs.iter().map(|(_, m)| *m).collect();
    trace.corners = pairs.iter().map(|(f, _)| f.center).collect();
    let (n_ok, extra) = locate_markers(&g, &lay, &mut anchors, &corner_mods, &cands, mod_px, trace);
    let corners_found = pairs.len() + extra;
    let Some(pos) = module_positions(n, &anchors) else { return err("géométrie incohérente") };
    trace.modules = pos.clone();

    // échantillonnage + égaliseur (retire la part « bavée » par les 4 voisins)
    let raw: Vec<f32> = pos.iter().map(|p| contrast.sample(p[0], p[1])).collect();
    let at = |x: isize, y: isize| raw[(y.clamp(0, n as isize - 1) as usize) * n + x.clamp(0, n as isize - 1) as usize];
    let mut vals = vec![0f32; n * n];
    for y in 0..n as isize {
        for x in 0..n as isize {
            let nb = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1);
            vals[y as usize * n + x as usize] = at(x, y) - 0.2 * nb;
        }
    }
    let mask: Vec<f32> = lay.kind.iter().map(|k| (*k == Kind::Data) as u8 as f32).collect();
    let local = masked_mean(&vals, &mask, n, 9);
    let centered: Vec<f32> = vals.iter().zip(&local).map(|(v, l)| v - l).collect();
    // normalisation sur fenêtre large : une zone uniforme (tache) garde une confiance faible
    let sq: Vec<f32> = centered.iter().map(|c| c * c).collect();
    let spread: Vec<f32> = masked_mean(&sq, &mask, n, 31).iter().map(|v| v.sqrt() + 1e-6).collect();

    let wh = whitening(lay.data_xy.len());
    let mut bits = Vec::with_capacity(lay.data_xy.len());
    let mut conf = Vec::with_capacity(lay.data_xy.len());
    for (i, &(x, y)) in lay.data_xy.iter().enumerate() {
        let j = y * n + x;
        let soft = centered[j] / spread[j];
        bits.push((soft < 0.0) as u8 ^ wh[i]);
        conf.push(soft.abs());
    }
    let perm = slot_permutation(lay.n_slots(), n);
    let mut stream = vec![0u8; perm.len()];
    let mut byte_conf = vec![0f32; perm.len()];
    for (i, &slot) in perm.iter().enumerate() {
        let mut b = 0u8;
        let mut c = f32::MAX;
        for k in 0..8 {
            b = (b << 1) | bits[slot * 8 + k];
            c = c.min(conf[slot * 8 + k]);
        }
        stream[i] = b;
        byte_conf[i] = c;
    }
    let plan = block_plan(lay.n_slots(), level);
    let total = lay.marker_centers.len();
    let Some((data, costs)) = rs_decode(&stream, &byte_conf, &plan) else {
        let costs: Vec<String> = plan.sizes.iter().scan(0usize, |pos, &sz| {
            let c = byte_conf[*pos..*pos + sz].iter().filter(|&&c| c < 0.3).count();
            *pos += sz;
            Some(c.to_string())
        }).collect();
        return err(format!("données irrécupérables ; coins {corners_found}, repères recalés {n_ok}/{total}, \
                            octets douteux par bloc [{}]", costs.join(",")));
    };
    let Some((codec, body)) = unpack_payload(&data) else { return err("CRC invalide") };
    Ok(DecodeResult { codec, body, n, level, corners_found, markers_found: n_ok, markers_total: total,
                      block_costs: costs, nsym: plan.nsym })
}
