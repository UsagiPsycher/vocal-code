//! Charge utile -> grille de modules -> image.

use crate::ecc::{block_plan, encode_format, pack_payload, rs_encode, CODEC_MIMI8};
use crate::layout::{layout, slot_permutation, whitening, FORMAT_SIDE, MAX_N, MIN_N, QUIET};
use image::GrayImage;

pub struct Grid {
    pub n: usize,
    pub cells: Vec<u8>, // 1 = noir
}

pub fn choose_size(payload_len: usize, level: u8) -> Option<usize> {
    (MIN_N..=MAX_N).find(|&n| block_plan(layout(n).n_slots(), level).data_capacity() >= payload_len)
}

pub fn build_grid(body: &[u8], level: u8) -> Result<Grid, String> {
    let payload = pack_payload(body, CODEC_MIMI8);
    let n = choose_size(payload.len(), level).ok_or("message trop long")?;
    let lay = layout(n);
    let plan = block_plan(lay.n_slots(), level);
    let mut data = payload;
    data.resize(plan.data_capacity(), 0);
    let stream = rs_encode(&data, &plan);

    let perm = slot_permutation(lay.n_slots(), n);
    let mut bits = vec![0u8; lay.data_xy.len()];
    for (i, &byte) in stream.iter().enumerate() {
        for b in 0..8 {
            bits[perm[i] * 8 + b] = (byte >> (7 - b)) & 1;
        }
    }
    for (b, w) in bits.iter_mut().zip(whitening(lay.data_xy.len())) {
        *b ^= w;
    }

    let mut cells = lay.fixed.clone();
    for (&(x, y), &b) in lay.data_xy.iter().zip(&bits) {
        cells[y * n + x] = b;
    }
    let fmt = encode_format(n, level);
    for &(ax, ay) in &lay.format_areas {
        for i in 0..FORMAT_SIDE * FORMAT_SIDE {
            let bit = (fmt[i / 8] >> (7 - i % 8)) & 1;
            cells[(ay + i / FORMAT_SIDE) * n + ax + i % FORMAT_SIDE] = bit;
        }
    }
    Ok(Grid { n, cells })
}

pub fn grid_to_image(g: &Grid, px_per_module: u32) -> GrayImage {
    let side = (g.n + 2 * QUIET) as u32 * px_per_module;
    GrayImage::from_fn(side, side, |x, y| {
        let mx = (x / px_per_module) as isize - QUIET as isize;
        let my = (y / px_per_module) as isize - QUIET as isize;
        let black = mx >= 0 && my >= 0 && (mx as usize) < g.n && (my as usize) < g.n
            && g.cells[my as usize * g.n + mx as usize] == 1;
        image::Luma([if black { 0 } else { 255 }])
    })
}

/// PNG en niveaux de gris avec la résolution d'impression (`dpi`) inscrite dans le fichier.
pub fn png_with_dpi(img: &GrayImage, dpi: f64) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, img.width(), img.height());
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let ppm = (dpi / 0.0254).round() as u32;
        enc.set_pixel_dims(Some(png::PixelDimensions { xppu: ppm, yppu: ppm, unit: png::Unit::Meter }));
        let mut w = enc.write_header().expect("en-tête PNG");
        w.write_image_data(img.as_raw()).expect("données PNG");
    }
    out
}
