//! Homographies (module -> image) et spline de plaque mince pour les déformations locales.

use nalgebra::{DMatrix, DVector, Matrix3, SymmetricEigen, SMatrix};

pub type Pt = [f64; 2];

#[derive(Clone, Copy, Debug)]
pub struct Homography(pub Matrix3<f64>);

impl Homography {
    pub fn apply(&self, p: Pt) -> Pt {
        let m = &self.0;
        let x = m[(0, 0)] * p[0] + m[(0, 1)] * p[1] + m[(0, 2)];
        let y = m[(1, 0)] * p[0] + m[(1, 1)] * p[1] + m[(1, 2)];
        let w = m[(2, 0)] * p[0] + m[(2, 1)] * p[1] + m[(2, 2)];
        [x / w, y / w]
    }
    /// Compose : d'abord `inner`, puis self.
    pub fn after(&self, inner: &Matrix3<f64>) -> Homography {
        Homography(self.0 * inner)
    }
}

fn normalizer(pts: &[Pt]) -> Matrix3<f64> {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let d = pts.iter().map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()).sum::<f64>() / n;
    let s = if d > 1e-12 { std::f64::consts::SQRT_2 / d } else { 1.0 };
    Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
}

/// Homographie par moindres carrés (DLT normalisée), >= 4 points.
pub fn fit_homography(src: &[Pt], dst: &[Pt]) -> Option<Homography> {
    fit_homography_weighted(src, dst, None)
}

/// Idem avec un poids par point (les points proches comptent davantage).
pub fn fit_homography_weighted(src: &[Pt], dst: &[Pt], weights: Option<&[f64]>) -> Option<Homography> {
    if src.len() < 4 {
        return None;
    }
    let (ts, td) = (normalizer(src), normalizer(dst));
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for (k, (s, d)) in src.iter().zip(dst).enumerate() {
        let wk = weights.map_or(1.0, |w| w[k] * w[k]);
        let a = ts * nalgebra::Vector3::new(s[0], s[1], 1.0);
        let b = td * nalgebra::Vector3::new(d[0], d[1], 1.0);
        let (x, y, u, v) = (a[0], a[1], b[0], b[1]);
        let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
        let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
        for r in [r1, r2] {
            for i in 0..9 {
                for j in 0..9 {
                    ata[(i, j)] += wk * r[i] * r[j];
                }
            }
        }
    }
    let eig = SymmetricEigen::new(ata);
    let (imin, _) = eig.eigenvalues.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?;
    let h = eig.eigenvectors.column(imin);
    let hn = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let m = td.try_inverse()? * hn * ts;
    if !m.iter().all(|v| v.is_finite()) || m[(2, 2)].abs() < 1e-15 {
        return None;
    }
    Some(Homography(m / m[(2, 2)]))
}

/// Affine exacte à partir de 3 points.
pub fn fit_affine(src: &[Pt], dst: &[Pt]) -> Option<Homography> {
    let a = Matrix3::new(src[0][0], src[1][0], src[2][0], src[0][1], src[1][1], src[2][1], 1.0, 1.0, 1.0);
    let b = Matrix3::new(dst[0][0], dst[1][0], dst[2][0], dst[0][1], dst[1][1], dst[2][1], 1.0, 1.0, 1.0);
    let m = b * a.try_inverse()?;
    Some(Homography(m))
}

/// Homographie si >= 4 points, sinon affine.
pub fn fit_map(src: &[Pt], dst: &[Pt]) -> Option<Homography> {
    if src.len() >= 4 {
        if let Some(h) = fit_homography(src, dst) {
            return Some(h);
        }
    }
    fit_affine(&src[..3], &dst[..3])
}

/// Spline de plaque mince 2D -> 2D (noyau r² ln r, terme affine), avec lissage.
pub struct Tps {
    centers: Vec<Pt>,
    w: Vec<Pt>,
    a: [[f64; 3]; 2],
}

fn phi(r2: f64) -> f64 {
    if r2 <= 0.0 { 0.0 } else { 0.5 * r2 * r2.ln() } // r² ln r = ½ r² ln r²
}

impl Tps {
    pub fn fit(centers: &[Pt], values: &[Pt], smoothing: f64) -> Option<Tps> {
        let n = centers.len();
        let m = n + 3;
        let mut k = DMatrix::<f64>::zeros(m, m);
        for i in 0..n {
            for j in 0..n {
                let dx = centers[i][0] - centers[j][0];
                let dy = centers[i][1] - centers[j][1];
                k[(i, j)] = phi(dx * dx + dy * dy);
            }
            k[(i, i)] += smoothing;
            let p = [1.0, centers[i][0], centers[i][1]];
            for c in 0..3 {
                k[(i, n + c)] = p[c];
                k[(n + c, i)] = p[c];
            }
        }
        let lu = k.lu();
        let mut w = vec![[0.0; 2]; n];
        let mut a = [[0.0; 3]; 2];
        for d in 0..2 {
            let mut rhs = DVector::<f64>::zeros(m);
            for i in 0..n {
                rhs[i] = values[i][d];
            }
            let sol = lu.solve(&rhs)?;
            for i in 0..n {
                w[i][d] = sol[i];
            }
            for c in 0..3 {
                a[d][c] = sol[n + c];
            }
        }
        Some(Tps { centers: centers.to_vec(), w, a })
    }

    pub fn eval(&self, p: Pt) -> Pt {
        let mut out = [0.0; 2];
        for d in 0..2 {
            out[d] = self.a[d][0] + self.a[d][1] * p[0] + self.a[d][2] * p[1];
        }
        for (c, w) in self.centers.iter().zip(&self.w) {
            let f = phi((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2));
            out[0] += w[0] * f;
            out[1] += w[1] * f;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn homographie_exacte() {
        let h = Homography(Matrix3::new(2.0, 0.3, 10.0, -0.2, 1.5, 5.0, 0.001, 0.0005, 1.0));
        let src: Vec<Pt> = vec![[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0], [50.0, 40.0]];
        let dst: Vec<Pt> = src.iter().map(|&p| h.apply(p)).collect();
        let f = fit_homography(&src, &dst).unwrap();
        for p in [[20.0, 70.0], [90.0, 10.0]] {
            let (a, b) = (h.apply(p), f.apply(p));
            assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
        }
    }
}
