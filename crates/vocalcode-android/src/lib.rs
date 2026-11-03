//! Pont JNI pour l'app Android (classe Kotlin `app.vocalcode.reader.Native`).
//!
//! - `load(weightsPath)` : charge le décodeur Mimi allégé ;
//! - `scan(gray, width, height, rowStride)` : cherche et lit un code dans une image en niveaux de gris ;
//!   renvoie `null` ou [usage f32][contour : 8 × f32][corps audio] (petit-boutiste) ;
//! - `decode(body, sink)` : décode le son en flux, `sink.onPcm(FloatArray): Boolean` reçoit chaque morceau.

use image::GrayImage;
use jni::objects::{JByteArray, JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jbyteArray, jint, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;
use std::sync::Mutex;
use vocalcode_audio::Decoder;
use vocalcode_core::{decode_image_traced, Trace};

static DECODER: Mutex<Option<Decoder>> = Mutex::new(None);

#[no_mangle]
pub extern "system" fn Java_app_vocalcode_reader_Native_load(mut env: JNIEnv, _: JClass, path: JString) -> jboolean {
    let Ok(path) = env.get_string(&path) else { return JNI_FALSE };
    let path: String = path.into();
    match Decoder::load(std::path::Path::new(&path)) {
        Ok(d) => {
            *DECODER.lock().unwrap() = Some(d);
            JNI_TRUE
        }
        Err(_) => JNI_FALSE,
    }
}

fn outline(t: &Trace, n: usize) -> [f32; 8] {
    let mut o = [0f32; 8];
    if t.modules.len() != n * n {
        return o;
    }
    let p = |x: usize, y: usize| t.modules[y * n + x];
    let ext = |c: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        [c[0] - 0.5 * (a[0] - c[0]) - 0.5 * (b[0] - c[0]), c[1] - 0.5 * (a[1] - c[1]) - 0.5 * (b[1] - c[1])]
    };
    let m = n - 1;
    let pts = [
        ext(p(0, 0), p(1, 0), p(0, 1)),
        ext(p(m, 0), p(m - 1, 0), p(m, 1)),
        ext(p(m, m), p(m - 1, m), p(m, m - 1)),
        ext(p(0, m), p(1, m), p(0, m - 1)),
    ];
    for (i, q) in pts.iter().enumerate() {
        o[2 * i] = q[0] as f32;
        o[2 * i + 1] = q[1] as f32;
    }
    o
}

#[no_mangle]
pub extern "system" fn Java_app_vocalcode_reader_Native_scan<'a>(
    env: JNIEnv<'a>, _: JClass, data: JByteArray<'a>, width: jint, height: jint, stride: jint,
) -> jbyteArray {
    let Ok(raw) = env.convert_byte_array(&data) else { return std::ptr::null_mut() };
    let (w, h, s) = (width as usize, height as usize, stride as usize);
    if w == 0 || h == 0 || s < w || raw.len() < s * (h - 1) + w {
        return std::ptr::null_mut();
    }
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        px.extend_from_slice(&raw[y * s..y * s + w]);
    }
    let Some(gray) = GrayImage::from_raw(w as u32, h as u32, px) else { return std::ptr::null_mut() };
    let mut trace = Trace::default();
    let Ok(r) = decode_image_traced(&gray, &mut trace) else { return std::ptr::null_mut() };
    if r.codec != vocalcode_core::ecc::CODEC_MIMI8 {
        return std::ptr::null_mut();
    }
    let mut out = Vec::with_capacity(36 + r.body.len());
    out.extend((r.worst_usage() as f32).to_le_bytes());
    for v in outline(&trace, r.n) {
        out.extend(v.to_le_bytes());
    }
    out.extend(&r.body);
    env.byte_array_from_slice(&out).map(|a| a.into_raw()).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_app_vocalcode_reader_Native_decode<'a>(
    mut env: JNIEnv<'a>, _: JClass, body: JByteArray<'a>, sink: JObject<'a>,
) -> jboolean {
    let Ok(body) = env.convert_byte_array(&body) else { return JNI_FALSE };
    let mut guard = DECODER.lock().unwrap();
    let Some(dec) = guard.as_mut() else { return JNI_FALSE };
    let res = dec.decode_stream(&body, 4, |pcm| {
        let Ok(arr) = env.new_float_array(pcm.len() as i32) else { return false };
        if env.set_float_array_region(&arr, 0, pcm).is_err() {
            return false;
        }
        let keep = env.call_method(&sink, "onPcm", "([F)Z", &[JValue::Object(&arr)])
            .and_then(|v| v.z())
            .unwrap_or(false);
        let _ = env.delete_local_ref(arr);
        keep
    });
    if res.is_ok() { JNI_TRUE } else { JNI_FALSE }
}
