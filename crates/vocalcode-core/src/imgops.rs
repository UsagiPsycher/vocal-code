//! Image en niveaux de gris flottants et opérations de base.

use image::GrayImage;

#[derive(Clone)]
pub struct FImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl FImage {
    pub fn new(w: usize, h: usize, v: f32) -> Self {
        Self { w, h, data: vec![v; w * h] }
    }
    pub fn from_gray(g: &GrayImage) -> Self {
        Self { w: g.width() as usize, h: g.height() as usize, data: g.as_raw().iter().map(|&v| v as f32).collect() }
    }
    pub fn to_gray(&self) -> GrayImage {
        GrayImage::from_raw(self.w as u32, self.h as u32,
                            self.data.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()).unwrap()
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }
    /// Échantillonnage bilinéaire ; coordonnées pixel (centre du pixel i = i), bord répété.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> f32 {
        let xf = (x.clamp(0.0, (self.w - 1) as f64)) as f32;
        let yf = (y.clamp(0.0, (self.h - 1) as f64)) as f32;
        let x0 = xf.floor() as usize;
        let y0 = yf.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let (fx, fy) = (xf - x0 as f32, yf - y0 as f32);
        let a = self.get(x0, y0) * (1.0 - fx) + self.get(x1, y0) * fx;
        let b = self.get(x0, y1) * (1.0 - fx) + self.get(x1, y1) * fx;
        a * (1.0 - fy) + b * fy
    }
    /// Échantillonnage avec valeur constante hors de l'image.
    #[inline]
    pub fn sample_or(&self, x: f64, y: f64, border: f32) -> f32 {
        if x < -0.5 || y < -0.5 || x > self.w as f64 - 0.5 || y > self.h as f64 - 0.5 {
            border
        } else {
            self.sample(x, y)
        }
    }
}

/// Moyenne sur une fenêtre carrée de côté `k` (impair), fenêtre tronquée aux bords.
pub fn box_mean(img: &FImage, k: usize) -> FImage {
    let r = (k / 2) as isize;
    let (w, h) = (img.w, img.h);
    let mut integ = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            row += img.data[y * w + x] as f64;
            integ[(y + 1) * (w + 1) + x + 1] = integ[y * (w + 1) + x + 1] + row;
        }
    }
    let mut out = FImage::new(w, h, 0.0);
    for y in 0..h as isize {
        let y0 = (y - r).max(0) as usize;
        let y1 = (y + r + 1).min(h as isize) as usize;
        for x in 0..w as isize {
            let x0 = (x - r).max(0) as usize;
            let x1 = (x + r + 1).min(w as isize) as usize;
            let s = integ[y1 * (w + 1) + x1] - integ[y0 * (w + 1) + x1] - integ[y1 * (w + 1) + x0]
                + integ[y0 * (w + 1) + x0];
            out.data[y as usize * w + x as usize] = (s / ((y1 - y0) * (x1 - x0)) as f64) as f32;
        }
    }
    out
}

/// Flou gaussien séparable, bord répété.
pub fn gaussian(img: &FImage, sigma: f32) -> FImage {
    if sigma <= 0.0 {
        return img.clone();
    }
    let r = (sigma * 3.0).ceil() as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let (w, h) = (img.w as isize, img.h as isize);
    let mut tmp = FImage::new(img.w, img.h, 0.0);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x + i as isize - r).clamp(0, w - 1);
                acc += kv * img.data[(y * w + xx) as usize];
            }
            tmp.data[(y * w + x) as usize] = acc;
        }
    }
    let mut out = FImage::new(img.w, img.h, 0.0);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let yy = (y + i as isize - r).clamp(0, h - 1);
                acc += kv * tmp.data[(yy * w + x) as usize];
            }
            out.data[(y * w + x) as usize] = acc;
        }
    }
    out
}

/// Moyenne masquée sur une grille n x n (fenêtre k impaire), pour le seuillage local des modules.
pub fn masked_mean(vals: &[f32], mask: &[f32], n: usize, k: usize) -> Vec<f32> {
    let num = FImage { w: n, h: n, data: vals.iter().zip(mask).map(|(v, m)| v * m).collect() };
    let den = FImage { w: n, h: n, data: mask.to_vec() };
    let (a, b) = (box_mean(&num, k), box_mean(&den, k));
    a.data.iter().zip(&b.data).map(|(x, y)| x / y.max(1e-6)).collect()
}
