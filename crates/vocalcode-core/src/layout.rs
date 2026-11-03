//! Disposition de la grille : repères, zones de format, modules de données.
//!
//! Conventions : grille N x N modules, (x = colonne, y = ligne), 1 = noir.
//! Le centre du module (x, y) a pour coordonnées continues (x + 0.5, y + 0.5).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub const QUIET: usize = 4;
pub const FINDER: usize = 7;
pub const MARKER: usize = 5;
pub const MARKER_SPACING: usize = 20;
pub const FORMAT_SIDE: usize = 8;
pub const MIN_N: usize = 41;
pub const MAX_N: usize = 1023;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Data,
    Finder,
    Format,
    Marker,
}

pub struct Layout {
    pub n: usize,
    pub kind: Vec<Kind>,             // n*n, ligne par ligne
    pub fixed: Vec<u8>,              // valeurs des modules non-données
    pub finder_origins: [(usize, usize); 4], // TL, TR, BR, BL (sens horaire)
    pub marker_centers: Vec<(usize, usize)>,
    pub format_areas: [(usize, usize); 4],
    pub data_xy: Vec<(usize, usize)>, // modules de données, ligne par ligne
}

impl Layout {
    pub fn n_slots(&self) -> usize {
        self.data_xy.len() / 8
    }
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.n + x
    }
    /// Centres des 4 repères de coin en coordonnées module, sens horaire depuis TL.
    pub fn corner_centers(n: usize) -> [(f64, f64); 4] {
        let a = 3.5;
        let b = n as f64 - 3.5;
        [(a, a), (b, a), (b, b), (a, b)]
    }
}

pub struct XorShift32(u32);

impl XorShift32 {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

pub fn finder_pattern(x: usize, y: usize) -> u8 {
    let d = x.min(y).min(6 - x).min(6 - y);
    (d != 1) as u8 // anneau extérieur noir, anneau blanc, carré central 3x3 noir
}

pub fn marker_pattern(x: usize, y: usize) -> u8 {
    let d = x.min(y).min(4 - x).min(4 - y);
    (d != 1) as u8
}

fn marker_axis(n: usize) -> Vec<usize> {
    let k = ((n - 5) as f64 / MARKER_SPACING as f64).ceil().max(1.0) as usize + 1;
    let (a, b) = (2.0, (n - 3) as f64);
    (0..k).map(|i| (a + (b - a) * i as f64 / (k - 1) as f64).round() as usize).collect()
}

fn build(n: usize) -> Layout {
    let mut kind = vec![Kind::Data; n * n];
    let mut fixed = vec![0u8; n * n];
    let origins = [(0, 0), (n - FINDER, 0), (n - FINDER, n - FINDER), (0, n - FINDER)];
    for &(ox, oy) in &origins {
        let (x0, x1) = (ox.saturating_sub(1), (ox + FINDER + 1).min(n));
        let (y0, y1) = (oy.saturating_sub(1), (oy + FINDER + 1).min(n));
        for y in y0..y1 {
            for x in x0..x1 {
                kind[y * n + x] = Kind::Finder;
                fixed[y * n + x] = 0;
            }
        }
        for y in 0..FINDER {
            for x in 0..FINDER {
                fixed[(oy + y) * n + ox + x] = finder_pattern(x, y);
            }
        }
    }
    let s = FORMAT_SIDE;
    let areas = [(9, 0), (0, 9), (n - s, 9), (9, n - s)]; // droite TL, sous TL, sous TR, droite BL
    for &(ax, ay) in &areas {
        for y in ay..ay + s {
            for x in ax..ax + s {
                kind[y * n + x] = Kind::Format;
            }
        }
    }
    let axis = marker_axis(n);
    let mut centers = Vec::new();
    for &cy in &axis {
        for &cx in &axis {
            let free = (cy - 2..=cy + 2).all(|y| (cx - 2..=cx + 2).all(|x| kind[y * n + x] == Kind::Data));
            if !free {
                continue;
            }
            for y in 0..MARKER {
                for x in 0..MARKER {
                    let i = (cy - 2 + y) * n + cx - 2 + x;
                    kind[i] = Kind::Marker;
                    fixed[i] = marker_pattern(x, y);
                }
            }
            centers.push((cx, cy));
        }
    }
    let mut data_xy = Vec::new();
    for y in 0..n {
        for x in 0..n {
            if kind[y * n + x] == Kind::Data {
                data_xy.push((x, y));
            }
        }
    }
    Layout { n, kind, fixed, finder_origins: origins, marker_centers: centers, format_areas: areas, data_xy }
}

pub fn layout(n: usize) -> Arc<Layout> {
    assert!((MIN_N..=MAX_N).contains(&n), "taille de grille hors limites : {n}");
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Layout>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(l) = cache.lock().unwrap().get(&n) {
        return l.clone();
    }
    let l = Arc::new(build(n));
    cache.lock().unwrap().insert(n, l.clone());
    l
}

/// Masque pseudo-aléatoire XOR appliqué aux modules de données.
pub fn whitening(n_bits: usize) -> Vec<u8> {
    let mut g = XorShift32::new(0x6C8E_9CF5);
    (0..n_bits).map(|_| (g.next() >> 31) as u8).collect()
}

/// perm[i] = emplacement (groupe de 8 modules de données) qui reçoit l'octet i du flux.
pub fn slot_permutation(n_slots: usize, n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n_slots).collect();
    let mut g = XorShift32::new(0x9E37_79B9 ^ n as u32);
    for i in (1..n_slots).rev() {
        let j = g.next() as usize % (i + 1);
        perm.swap(i, j);
    }
    perm
}
