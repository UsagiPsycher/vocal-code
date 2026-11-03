mod bench;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;
use vocalcode_core::layout::QUIET;
use vocalcode_core::{build_grid, decode_image, grid_to_image};

#[derive(Parser)]
#[command(name = "vocalcode", about = "Messages vocaux dans un code 2D imprimable")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Audio (mp3, m4a, wav…) ou .bin déjà encodé -> image du code (PNG prête à imprimer)
    Encode {
        input: PathBuf,
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Durée maximale gardée, en secondes
        #[arg(long, default_value_t = 30.0)]
        max_secondes: f32,
        /// Niveau de correction : 0=20 %, 1=30 %, 2=40 %, 3=50 % de parité
        #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(0..=3))]
        ecc: u8,
        /// Largeur imprimée en cm, marge blanche incluse
        #[arg(long, default_value_t = 8.0)]
        largeur_cm: f64,
        #[arg(long, default_value_t = 12)]
        px_module: u32,
    },
    /// Photo ou image d'un code -> fichier audio
    Decode {
        image: PathBuf,
        #[arg(short, long, default_value = "message.wav")]
        out: PathBuf,
        /// Enregistre aussi les octets bruts décodés
        #[arg(long)]
        bin: Option<PathBuf>,
    },
    /// Diagnostic : superpose la géométrie détectée sur l'image
    #[command(hide = true)]
    Debug {
        image: PathBuf,
        #[arg(short, long, default_value = "debug.png")]
        out: PathBuf,
    },
    /// Simule une photo abîmée d'une image de code (scénario du banc, par numéro)
    #[command(hide = true)]
    Simule {
        image: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        #[arg(long, default_value_t = 13)]
        scenario: usize,
        #[arg(long, default_value_t = 1)]
        graine: u64,
    },
    /// Banc de robustesse sur images dégradées
    Bench {
        /// Fichier .bin (corps audio encodé)
        bin: PathBuf,
        #[arg(long, default_value_t = 8)]
        essais: usize,
        #[arg(long, default_value_t = 2)]
        ecc: u8,
        /// Dossier où enregistrer une image dégradée par scénario
        #[arg(long)]
        exemples: Option<PathBuf>,
        /// Ne lancer que les scénarios contenant ce texte
        #[arg(long)]
        seul: Option<String>,
    },
}

fn write_png(path: &Path, img: &image::GrayImage, dpi: f64) -> Result<()> {
    std::fs::write(path, vocalcode_core::render::png_with_dpi(img, dpi))?;
    Ok(())
}

fn load_photo(path: &Path) -> Result<image::GrayImage> {
    use image::{DynamicImage, ImageDecoder, ImageReader};
    let mut dec = ImageReader::open(path)?.with_guessed_format()?.into_decoder()
        .with_context(|| format!("image illisible : {}", path.display()))?;
    let orientation = dec.orientation()?;
    let mut img = DynamicImage::from_decoder(dec)?;
    img.apply_orientation(orientation);
    Ok(img.to_luma8())
}

fn wait_enter() {
    println!("\nAppuie sur Entrée pour fermer.");
    let _ = std::io::stdin().read_line(&mut String::new());
}

/// Glisser-déposer d'images sur l'exécutable (ou double-clic sans argument).
fn drop_mode(files: &[PathBuf]) -> bool {
    if files.is_empty() {
        println!("vocalcode — décodeur de messages vocaux\n");
        println!("Glisse une ou plusieurs photos de code sur vocalcode.exe :");
        println!("le message est décodé, enregistré en .wav à côté de la photo, puis joué.");
        println!("\n(En ligne de commande : vocalcode --help)");
        wait_enter();
        return true;
    }
    let mut codec = None;
    for f in files {
        println!("{}", f.display());
        let res = (|| -> Result<PathBuf> {
            let gray = load_photo(f)?;
            let t0 = Instant::now();
            let r = decode_image(&gray)?;
            println!("  décodé en {:.0} ms ({} octets, correction utilisée {:.0}%)",
                     t0.elapsed().as_secs_f64() * 1000.0, r.body.len(), r.worst_usage() * 100.0);
            if codec.is_none() {
                println!("  chargement du modèle audio…");
                codec = Some(vocalcode_audio::Codec::load()?);
            }
            let pcm = codec.as_mut().unwrap().decode(&r.body)?;
            let out = f.with_extension("wav");
            vocalcode_audio::write_wav(&out, &pcm, vocalcode_audio::SAMPLE_RATE)?;
            Ok(out)
        })();
        match res {
            Ok(out) => {
                println!("  -> {}", out.display());
                let _ = std::process::Command::new("cmd").args(["/C", "start", ""]).arg(&out).spawn();
            }
            Err(e) => println!("  échec : {e}"),
        }
    }
    wait_enter();
    true
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let is_cmd = args.first().is_some_and(|a| a.starts_with('-')
        || ["encode", "decode", "bench", "debug", "simule", "help"].contains(&a.as_str()));
    if !is_cmd {
        let files: Vec<PathBuf> = args.iter().map(PathBuf::from).collect();
        if files.iter().all(|f| f.is_file()) && drop_mode(&files) {
            return Ok(());
        }
    }
    match Cli::parse().cmd {
        Cmd::Encode { input, out, max_secondes, ecc, largeur_cm, px_module } => {
            let out = out.unwrap_or_else(|| input.with_extension("png"));
            let (body, dur) = if input.extension().is_some_and(|e| e == "bin") {
                (std::fs::read(&input)?, None)
            } else {
                let pcm = vocalcode_audio::load_audio(&input, max_secondes)?;
                let mut codec = vocalcode_audio::Codec::load()?;
                let body = codec.encode(&pcm)?;
                std::fs::write(out.with_extension("bin"), &body)?;
                (body, Some(pcm.len() as f32 / vocalcode_audio::SAMPLE_RATE as f32))
            };
            let grid = build_grid(&body, ecc).map_err(anyhow::Error::msg)?;
            let img = grid_to_image(&grid, px_module);
            write_png(&out, &img, img.width() as f64 / (largeur_cm / 2.54))?;
            let module_mm = largeur_cm * 10.0 / (grid.n + 2 * QUIET) as f64;
            let d = dur.map(|d| format!("{d:.1} s -> ")).unwrap_or_default();
            println!("{d}{} octets, grille {}x{}, module {module_mm:.2} mm pour {largeur_cm} cm -> {}",
                     body.len(), grid.n, grid.n, out.display());
        }
        Cmd::Decode { image, out, bin } => {
            let gray = load_photo(&image)?;
            let t0 = Instant::now();
            let r = decode_image(&gray)?;
            println!("grille {0}x{0}, coins {1}/4, repères {2}/{3}, correction utilisée (pire bloc) {4:.0}%, {5} octets en {6:.0} ms",
                     r.n, r.corners_found, r.markers_found, r.markers_total, r.worst_usage() * 100.0, r.body.len(),
                     t0.elapsed().as_secs_f64() * 1000.0);
            if let Some(b) = bin {
                std::fs::write(b, &r.body)?;
            }
            if r.codec != vocalcode_core::ecc::CODEC_MIMI8 {
                bail!("codec inconnu : {}", r.codec);
            }
            let mut codec = vocalcode_audio::Codec::load()?;
            let pcm = codec.decode(&r.body)?;
            vocalcode_audio::write_wav(&out, &pcm, vocalcode_audio::SAMPLE_RATE)?;
            println!("-> {}", out.display());
        }
        Cmd::Debug { image, out } => {
            let gray = load_photo(&image)?;
            let mut tr = vocalcode_core::Trace::default();
            match vocalcode_core::decode_image_traced(&gray, &mut tr) {
                Ok(r) => println!("OK, correction {:.0}%", r.worst_usage() * 100.0),
                Err(e) => println!("échec : {e}"),
            }
            let mut rgb = image::DynamicImage::ImageLuma8(gray).to_rgb8();
            let mut dot = |p: [f64; 2], c: [u8; 3], r: i64| {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (x, y) = (p[0].round() as i64 + dx, p[1].round() as i64 + dy);
                        if x >= 0 && y >= 0 && (x as u32) < rgb.width() && (y as u32) < rgb.height() {
                            rgb.put_pixel(x as u32, y as u32, image::Rgb(c));
                        }
                    }
                }
            };
            for p in &tr.modules {
                dot(*p, [0, 160, 255], 0);
            }
            for (pred, found, score) in &tr.markers {
                if *score >= 0.45 {
                    dot(*found, [0, 220, 0], 2);
                } else {
                    dot(*pred, [255, 0, 0], 3);
                }
            }
            for c in &tr.corners {
                dot(*c, [255, 0, 255], 4);
            }
            rgb.save(&out)?;
            println!("-> {}", out.display());
        }
        Cmd::Simule { image, out, scenario, graine } => {
            let src = load_photo(&image)?;
            // ramène l'image à HI px/module (marge blanche comprise) avant de la dégrader
            let n_px = src.width();
            let cells = (n_px as f64 / 12.0).round() as u32; // les PNG générés font 12 px/module
            let base = image::imageops::resize(&src, cells * bench::HI, cells * bench::HI, image::imageops::FilterType::Nearest);
            let (name, p) = bench::scenarios().into_iter().nth(scenario).context("scénario inconnu")?;
            bench::degrade_framed(&base, &p, graine).save(&out)?;
            println!("{name} -> {}", out.display());
        }
        Cmd::Bench { bin, essais, ecc, exemples, seul } => {
            let body = std::fs::read(&bin)?;
            bench::run(&body, ecc, essais, exemples.as_deref(), seul.as_deref())?;
        }
    }
    Ok(())
}
