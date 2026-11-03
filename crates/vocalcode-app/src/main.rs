#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::io::Cursor;
use std::sync::{Condvar, Mutex};
use tauri::ipc::{InvokeBody, Request, Response};
use vocalcode_audio::{envelope, wav_bytes, Codec, SAMPLE_RATE};
use vocalcode_core::layout::QUIET;
use vocalcode_core::render::png_with_dpi;
use vocalcode_core::{build_grid, decode_image_traced, grid_to_image, Grid, Trace};

const MAX_SECONDS: f32 = 30.0;
const ENV_STEP: f32 = 0.1; // une barre de la bande = 100 ms
const ENV_BUCKETS: usize = 300;
const ECC_LEVEL: u8 = 2;
const PREVIEW_PX: u32 = 8;
const EXPORT_PX: u32 = 12;

// ---------- modèle audio (chargé en arrière-plan au démarrage) ----------

enum Model {
    Loading,
    Ready(Codec),
    Failed(String),
}

static MODEL: Mutex<Model> = Mutex::new(Model::Loading);
static MODEL_CV: Condvar = Condvar::new();

fn load_model_in_background() {
    std::thread::spawn(|| {
        let m = match Codec::load() {
            Ok(c) => Model::Ready(c),
            Err(e) => Model::Failed(format!("Modèle audio indisponible : {e}")),
        };
        *MODEL.lock().unwrap() = m;
        MODEL_CV.notify_all();
    });
}

fn with_codec<T>(f: impl FnOnce(&mut Codec) -> Result<T, String>) -> Result<T, String> {
    let mut g = MODEL.lock().unwrap();
    while matches!(*g, Model::Loading) {
        g = MODEL_CV.wait(g).unwrap();
    }
    match &mut *g {
        Model::Ready(c) => f(c),
        Model::Failed(e) => Err(e.clone()),
        Model::Loading => unreachable!(),
    }
}

#[tauri::command]
fn model_status() -> String {
    match &*MODEL.lock().unwrap() {
        Model::Loading => "loading".into(),
        Model::Ready(_) => "ready".into(),
        Model::Failed(e) => e.clone(),
    }
}

// ---------- état de la session ----------

#[derive(Default)]
struct Session {
    grid: Option<Grid>,
    preview_png: Vec<u8>,
    audio_wav: Vec<u8>,
}

static SESSION: Mutex<Option<Session>> = Mutex::new(None);

fn session<T>(f: impl FnOnce(&mut Session) -> T) -> T {
    let mut g = SESSION.lock().unwrap();
    f(g.get_or_insert_with(Session::default))
}

fn raw_body(req: &Request<'_>) -> Result<Vec<u8>, String> {
    match req.body() {
        InvokeBody::Raw(b) => Ok(b.clone()),
        _ => Err("corps binaire attendu".into()),
    }
}

fn header(req: &Request<'_>, name: &str) -> Option<String> {
    req.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

// ---------- Créer ----------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EncodeInfo {
    n: usize,
    seconds: f32,
    bytes: usize,
    truncated: bool,
    envelope: Vec<f32>,
}

/// Corps : PCM f32 LE mono 24 kHz (`x-kind: pcm`) ou fichier audio (`x-kind: file`, `x-ext`).
#[tauri::command]
async fn encode_audio(request: Request<'_>) -> Result<EncodeInfo, String> {
    let body = raw_body(&request)?;
    let kind = header(&request, "x-kind").unwrap_or_default();
    let ext = header(&request, "x-ext");
    blocking(move || {
        let (pcm, truncated) = if kind == "pcm" {
            let mut pcm: Vec<f32> = body.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let max = (MAX_SECONDS * SAMPLE_RATE as f32) as usize;
            let t = pcm.len() > max;
            pcm.truncate(max);
            (pcm, t)
        } else {
            // on lit une seconde de plus pour savoir si le fichier a été coupé
            let pcm = vocalcode_audio::load_audio_bytes(body, ext.as_deref(), MAX_SECONDS + 1.0)
                .map_err(|_| "Format audio non reconnu. Essaie un fichier mp3, m4a, wav, flac ou ogg.".to_string())?;
            let max = (MAX_SECONDS * SAMPLE_RATE as f32) as usize;
            let t = pcm.len() > max;
            (pcm.into_iter().take(max).collect(), t)
        };
        if pcm.len() < SAMPLE_RATE as usize / 4 {
            return Err("Message trop court.".into());
        }
        let (code_body, heard) = with_codec(|c| {
            let b = c.encode(&pcm).map_err(|e| e.to_string())?;
            let heard = c.decode(&b).map_err(|e| e.to_string())?;
            Ok((b, heard))
        })?;
        let grid = build_grid(&code_body, ECC_LEVEL)?;
        let preview = png_with_dpi(&grid_to_image(&grid, PREVIEW_PX), 300.0);
        let wav = wav_bytes(&heard, SAMPLE_RATE).map_err(|e| e.to_string())?;
        let info = EncodeInfo {
            n: grid.n,
            seconds: pcm.len() as f32 / SAMPLE_RATE as f32,
            bytes: code_body.len(),
            truncated,
            envelope: envelope(&heard, SAMPLE_RATE, ENV_STEP, ENV_BUCKETS),
        };
        session(|s| {
            s.grid = Some(grid);
            s.preview_png = preview;
            s.audio_wav = wav;
        });
        Ok(info)
    }).await
}

#[tauri::command]
fn take_preview() -> Response {
    Response::new(session(|s| s.preview_png.clone()))
}

#[tauri::command]
fn take_audio() -> Response {
    Response::new(session(|s| s.audio_wav.clone()))
}

#[tauri::command]
async fn save_png(path: String, width_cm: f64) -> Result<(), String> {
    blocking(move || {
        let png = session(|s| {
            s.grid.as_ref().map(|g| {
                let img = grid_to_image(g, EXPORT_PX);
                let dpi = img.width() as f64 / (width_cm / 2.54);
                png_with_dpi(&img, dpi)
            })
        }).ok_or("Aucun code à enregistrer.")?;
        std::fs::write(&path, png).map_err(|e| format!("Enregistrement impossible : {e}"))
    }).await
}

#[tauri::command]
async fn save_wav(path: String) -> Result<(), String> {
    blocking(move || {
        let wav = session(|s| s.audio_wav.clone());
        if wav.is_empty() {
            return Err("Aucun son à enregistrer.".into());
        }
        std::fs::write(&path, wav).map_err(|e| format!("Enregistrement impossible : {e}"))
    }).await
}

#[tauri::command]
fn quiet_modules() -> usize {
    QUIET
}

// ---------- Lire ----------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DecodeInfo {
    width: u32,
    height: u32,
    outline: Vec<[f64; 2]>,
    seconds: f32,
    usage: f64,
    envelope: Vec<f32>,
}

fn load_photo(bytes: Vec<u8>) -> Result<image::GrayImage, String> {
    use image::{DynamicImage, ImageDecoder, ImageReader};
    let bad = || "Image illisible. Utilise une photo JPEG, PNG ou WebP.".to_string();
    let mut dec = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|_| bad())?
        .into_decoder().map_err(|_| bad())?;
    let orientation = dec.orientation().map_err(|_| bad())?;
    let mut img = DynamicImage::from_decoder(dec).map_err(|_| bad())?;
    img.apply_orientation(orientation);
    Ok(img.to_luma8())
}

fn friendly(e: &str) -> String {
    if e.contains("coin") {
        "Code introuvable. Cadre la photo pour que les quatre coins soient visibles.".into()
    } else {
        "Code trop abîmé ou flou. Reprends la photo de plus près, sans reflet.".into()
    }
}

/// Contour extérieur du code, à partir des positions des modules des coins.
fn outline(t: &Trace, n: usize) -> Vec<[f64; 2]> {
    if t.modules.len() != n * n {
        return vec![];
    }
    let p = |x: usize, y: usize| t.modules[y * n + x];
    let ext = |c: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        // recule d'un demi-module dans les deux directions opposées aux voisins
        [c[0] - 0.5 * (a[0] - c[0]) - 0.5 * (b[0] - c[0]), c[1] - 0.5 * (a[1] - c[1]) - 0.5 * (b[1] - c[1])]
    };
    let m = n - 1;
    vec![
        ext(p(0, 0), p(1, 0), p(0, 1)),
        ext(p(m, 0), p(m - 1, 0), p(m, 1)),
        ext(p(m, m), p(m - 1, m), p(m, m - 1)),
        ext(p(0, m), p(1, m), p(0, m - 1)),
    ]
}

#[tauri::command]
async fn decode_photo(request: Request<'_>) -> Result<DecodeInfo, String> {
    let body = raw_body(&request)?;
    blocking(move || {
        let gray = load_photo(body)?;
        let mut trace = Trace::default();
        let r = decode_image_traced(&gray, &mut trace).map_err(|e| friendly(&e.0))?;
        let pcm = with_codec(|c| c.decode(&r.body).map_err(|e| e.to_string()))?;
        let wav = wav_bytes(&pcm, SAMPLE_RATE).map_err(|e| e.to_string())?;
        session(|s| s.audio_wav = wav);
        Ok(DecodeInfo {
            width: gray.width(),
            height: gray.height(),
            outline: outline(&trace, r.n),
            seconds: pcm.len() as f32 / SAMPLE_RATE as f32,
            usage: r.worst_usage(),
            envelope: envelope(&pcm, SAMPLE_RATE, ENV_STEP, ENV_BUCKETS),
        })
    }).await
}

fn initial_path() -> Option<std::path::PathBuf> {
    std::env::args_os().nth(1).map(std::path::PathBuf::from).filter(|p| p.is_file())
}

#[tauri::command]
fn initial_file_name() -> Option<String> {
    initial_path().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
}

#[tauri::command]
fn initial_file() -> Response {
    Response::new(initial_path().and_then(|p| std::fs::read(p).ok()).unwrap_or_default())
}

fn main() {
    load_model_in_background();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            model_status, encode_audio, take_preview, take_audio, save_png, save_wav, quiet_modules, decode_photo,
            initial_file_name, initial_file
        ])
        .run(tauri::generate_context!())
        .expect("démarrage de l'application");
}
