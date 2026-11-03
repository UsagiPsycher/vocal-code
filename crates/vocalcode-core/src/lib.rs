//! vocal-code : messages vocaux dans un code 2D imprimable.
//!
//! Ce crate ne dépend ni de l'audio ni du système : il pourra être compilé tel quel pour Android.

pub mod detect;
pub mod ecc;
pub mod geom;
pub mod imgops;
pub mod layout;
pub mod render;
pub mod rs;

pub use detect::{decode_image, decode_image_traced, DecodeError, DecodeResult, Trace};
pub use render::{build_grid, grid_to_image, Grid};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aller_retour_image_propre() {
        let body: Vec<u8> = (0..4501u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let g = build_grid(&body, 2).unwrap();
        assert_eq!(g.n, 256);
        for ppm in [3, 5] {
            let img = grid_to_image(&g, ppm);
            let r = decode_image(&img).expect("décodage");
            assert_eq!(r.body, body);
        }
    }
}
