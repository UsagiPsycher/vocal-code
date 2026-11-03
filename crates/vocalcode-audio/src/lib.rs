//! Audio <-> octets via le codec neuronal Mimi (Kyutai), 8 codebooks, ~1,1 kbps.
//!
//! Format du corps : nb de trames (u16 big-endian) puis les codes, trame par trame,
//! 8 codes de 11 bits chacun, bits de poids fort en premier.

use anyhow::{bail, Result};
#[cfg(feature = "io")]
use anyhow::Context;
use candle_core::{Device, Tensor};
#[cfg(feature = "io")]
use std::fs::File;
#[cfg(feature = "io")]
use std::io::{Read, Write};
#[cfg(feature = "io")]
use std::path::Path;
#[cfg(feature = "io")]
use std::path::PathBuf;

pub const SAMPLE_RATE: u32 = 24_000;
pub const N_Q: usize = 8;
pub const BITS: usize = 11;
pub const FRAME_RATE: f32 = 12.5;

/// Poids Mimi complets (Kyutai, CC-BY 4.0). La version est figée : les codes imprimés
/// dépendent de ces poids exacts, un autre modèle exigerait un nouvel identifiant de codec.
#[cfg(feature = "io")]
pub const WEIGHTS_FILE: &str = "mimi-e351c8d8-checkpoint125.safetensors";
#[cfg(feature = "io")]
pub const WEIGHTS_SHA256: &str = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50";
/// Sources, essayées dans l'ordre : Hugging Face à une révision figée, puis le miroir du projet.
#[cfg(feature = "io")]
pub const WEIGHTS_URLS: [&str; 2] = [
    "https://huggingface.co/kyutai/moshiko-candle-bf16/resolve/50c436c9849a20685a47f087dc243f6bddac207b/tokenizer-e351c8d8-checkpoint125.safetensors",
    "https://github.com/UsagiPsycher/vocal-code/releases/download/modele-mimi-v1/mimi-e351c8d8-checkpoint125.safetensors",
];

#[cfg(feature = "io")]
pub struct Codec {
    mimi: moshi::mimi::Mimi,
    dev: Device,
}

#[cfg(feature = "io")]
fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("VOCALCODE_CACHE") {
        return PathBuf::from(d);
    }
    let base = std::env::var("LOCALAPPDATA")
        .or_else(|_| std::env::var("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".cache"));
    base.join("vocalcode")
}

/// Chemin des poids Mimi, téléchargés et vérifiés (SHA-256) au premier usage.
#[cfg(feature = "io")]
pub fn weights_path() -> Result<PathBuf> {
    let dir = cache_dir();
    let path = dir.join(WEIGHTS_FILE);
    if path.exists() {
        return Ok(path);
    }
    std::fs::create_dir_all(&dir)?;
    eprintln!("téléchargement des poids Mimi (385 Mo, une seule fois) -> {}", path.display());
    let tmp = path.with_extension("part");
    let mut last_err = None;
    for url in WEIGHTS_URLS {
        match download_verified(url, &tmp, WEIGHTS_SHA256) {
            Ok(()) => {
                std::fs::rename(&tmp, &path)?;
                return Ok(path);
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                last_err = Some(e.context(format!("source {url}")));
            }
        }
    }
    Err(last_err.unwrap().context("poids Mimi indisponibles (connexion nécessaire au premier lancement)"))
}

#[cfg(feature = "io")]
fn download_verified(url: &str, dst: &Path, sha256: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let resp = ureq::get(url).call()?;
    let mut reader = resp.into_reader();
    let mut out = std::io::BufWriter::new(File::create(dst)?);
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        out.write_all(&buf[..n])?;
    }
    out.flush()?;
    let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if got != sha256 {
        bail!("empreinte inattendue ({got})");
    }
    Ok(())
}

#[cfg(feature = "io")]
impl Codec {
    pub fn load() -> Result<Self> {
        let dev = Device::Cpu;
        let w = weights_path()?;
        let mimi = moshi::mimi::load(w.to_str().context("chemin non UTF-8")?, Some(N_Q), &dev)?;
        Ok(Self { mimi, dev })
    }

    /// PCM mono 24 kHz -> corps binaire.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u8>> {
        self.mimi.reset_state();
        let x = Tensor::from_slice(pcm, (1, 1, pcm.len()), &self.dev)?;
        let codes = self.mimi.encode(&x)?; // (1, N_Q, T)
        let codes = codes.squeeze(0)?.to_dtype(candle_core::DType::U32)?.to_vec2::<u32>()?;
        let t = codes[0].len();
        let mut w = BitWriter::default();
        for f in 0..t {
            for q in codes.iter().take(N_Q) {
                w.put(q[f], BITS);
            }
        }
        let mut out = (t as u16).to_be_bytes().to_vec();
        out.extend(w.finish());
        Ok(out)
    }

    /// Corps binaire -> PCM mono 24 kHz.
    pub fn decode(&mut self, body: &[u8]) -> Result<Vec<f32>> {
        let codes = unpack_codes(body, &self.dev)?;
        self.mimi.reset_state();
        let pcm = self.mimi.decode(&codes)?;
        Ok(pcm.flatten_all()?.to_vec1::<f32>()?)
    }
}

fn unpack_codes(body: &[u8], dev: &Device) -> Result<Tensor> {
    if body.len() < 2 {
        bail!("corps audio vide");
    }
    let t = u16::from_be_bytes([body[0], body[1]]) as usize;
    if (body.len() - 2) * 8 < t * N_Q * BITS {
        bail!("corps audio tronqué");
    }
    let mut r = BitReader::new(&body[2..]);
    let mut codes = vec![0u32; N_Q * t];
    for f in 0..t {
        for q in 0..N_Q {
            codes[q * t + f] = r.get(BITS);
        }
    }
    Ok(Tensor::from_vec(codes, (1, N_Q, t), dev)?)
}

/// Décodeur Mimi seul (lecteur) : ne charge que les poids nécessaires au décodage,
/// cf. tools/mimi_decoder_weights.py (~99 Mo au lieu de 385).
pub struct Decoder {
    quantizer: moshi::quantization::SplitResidualVectorQuantizer,
    upsample: moshi::conv::ConvTrUpsample1d,
    transformer: moshi::transformer::Transformer,
    decoder: moshi::seanet::SeaNetDecoder,
    dev: Device,
}

impl Decoder {
    pub fn load(weights: &std::path::Path) -> Result<Self> {
        use moshi::streaming::StreamingModule as _;
        let dev = Device::Cpu;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], candle_core::DType::F32, &dev)?
        };
        let cfg = moshi::mimi::Config::v0_1(Some(N_Q));
        let dim = cfg.seanet.dimension;
        let quantizer = moshi::quantization::SplitResidualVectorQuantizer::new(
            cfg.quantizer_dim, Some(dim), Some(dim), cfg.quantizer_n_q, cfg.quantizer_bins, vb.pp("quantizer"))?;
        let encoder_frame_rate = cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;
        let stride = (encoder_frame_rate / cfg.frame_rate) as usize;
        let upsample = moshi::conv::ConvTrUpsample1d::new(stride, dim, true, true, vb.pp("upsample"))?;
        let mut transformer = moshi::transformer::Transformer::new(None, dim, &cfg.transformer, vb.pp("decoder_transformer"))?;
        transformer.reset_state();
        let decoder = moshi::seanet::SeaNetDecoder::new(&cfg.seanet, vb.pp("decoder"))?;
        Ok(Self { quantizer, upsample, transformer, decoder, dev })
    }

    /// Durée du message (s) d'après l'en-tête du corps.
    pub fn duration(body: &[u8]) -> f32 {
        if body.len() < 2 { 0.0 } else { u16::from_be_bytes([body[0], body[1]]) as f32 / FRAME_RATE }
    }

    /// Décodage en flux : `on_pcm` reçoit chaque morceau de son dès qu'il est prêt
    /// (renvoyer `false` pour arrêter). `chunk` = nombre de trames Mimi par morceau (80 ms chacune).
    pub fn decode_stream(&mut self, body: &[u8], chunk: usize, mut on_pcm: impl FnMut(&[f32]) -> bool) -> Result<()> {
        use moshi::streaming::{StreamMask, StreamTensor, StreamingModule as _};
        let codes = unpack_codes(body, &self.dev)?;
        let t = codes.dim(2)?;
        self.upsample.reset_state();
        self.transformer.reset_state();
        self.decoder.reset_state();
        let m = StreamMask::empty();
        let mut f = 0;
        while f < t {
            let k = chunk.min(t - f);
            let c = codes.narrow(2, f, k)?;
            let emb = StreamTensor::from_tensor(self.quantizer.decode(&c)?);
            let x = self.upsample.step(&emb, &m)?;
            let x = self.transformer.step(&x, &m)?;
            let x = self.decoder.step(&x, &m)?;
            if let Some(pcm) = x.as_option() {
                if !on_pcm(&pcm.flatten_all()?.to_vec1::<f32>()?) {
                    break;
                }
            }
            f += k;
        }
        Ok(())
    }

    /// Corps binaire -> PCM mono 24 kHz.
    pub fn decode(&mut self, body: &[u8]) -> Result<Vec<f32>> {
        use candle_core::Module;
        use moshi::streaming::StreamingModule as _;
        let codes = unpack_codes(body, &self.dev)?;
        let emb = self.quantizer.decode(&codes)?;
        let emb = emb.apply(&self.upsample)?;
        self.transformer.reset_state();
        let outs = self.transformer.forward(&emb)?;
        let pcm = self.decoder.forward(&outs[0])?;
        Ok(pcm.flatten_all()?.to_vec1::<f32>()?)
    }
}

#[cfg(feature = "io")]
#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: usize,
}

#[cfg(feature = "io")]
impl BitWriter {
    fn put(&mut self, v: u32, bits: usize) {
        self.acc = (self.acc << bits) | (v as u64 & ((1 << bits) - 1));
        self.n += bits;
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push((self.acc << (8 - self.n)) as u8);
        }
        self.out
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn get(&mut self, bits: usize) -> u32 {
        let mut v = 0u32;
        for _ in 0..bits {
            let b = (self.data[self.pos / 8] >> (7 - self.pos % 8)) & 1;
            v = (v << 1) | b as u32;
            self.pos += 1;
        }
        v
    }
}

// ---------- Fichiers audio ----------

#[cfg(feature = "io")]
/// Lit un fichier audio (mp3, m4a/aac, wav, flac, ogg) en mono 24 kHz, tronqué à `max_seconds`.
pub fn load_audio(path: &Path, max_seconds: f32) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("ouverture de {}", path.display()))?;
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_owned);
    decode_source(Box::new(file), ext.as_deref(), max_seconds)
}

#[cfg(feature = "io")]
/// Idem depuis des octets en mémoire ; `ext` aide à reconnaître le format.
pub fn load_audio_bytes(bytes: Vec<u8>, ext: Option<&str>, max_seconds: f32) -> Result<Vec<f32>> {
    decode_source(Box::new(std::io::Cursor::new(bytes)), ext, max_seconds)
}

#[cfg(feature = "io")]
fn decode_source(src: Box<dyn symphonia::core::io::MediaSource>, ext: Option<&str>, max_seconds: f32) -> Result<Vec<f32>> {
    use symphonia::core::{audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions,
                          io::MediaSourceStream, meta::MetadataOptions, probe::Hint};
    let mss = MediaSourceStream::new(src, Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = ext {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("format audio non reconnu")?;
    let mut format = probed.format;
    let track = format.default_track().context("aucune piste audio")?.clone();
    let mut dec = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let src_rate = track.codec_params.sample_rate.context("fréquence inconnue")?;
    let max_src = (max_seconds * src_rate as f32) as usize;

    let mut mono = Vec::new();
    while mono.len() < max_src {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track.id {
            continue;
        }
        let decoded = match dec.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let ch = decoded.spec().channels.count();
        let mut sb = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sb.copy_interleaved_ref(decoded);
        for frame in sb.samples().chunks(ch) {
            mono.push(frame.iter().sum::<f32>() / ch as f32);
        }
    }
    mono.truncate(max_src);
    resample(&mono, src_rate, SAMPLE_RATE)
}

#[cfg(feature = "io")]
pub fn resample(x: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};
    if from == to {
        return Ok(x.to_vec());
    }
    let chunk = 1024;
    let mut rs = FftFixedIn::<f32>::new(from as usize, to as usize, chunk, 2, 1)?;
    let delay = rs.output_delay();
    let mut out = Vec::with_capacity(x.len() * to as usize / from as usize + chunk);
    let mut pos = 0;
    while pos + chunk <= x.len() {
        out.extend(&rs.process(&[&x[pos..pos + chunk]], None)?[0]);
        pos += chunk;
    }
    let mut tail = x[pos..].to_vec();
    tail.resize(chunk, 0.0);
    out.extend(&rs.process(&[&tail], None)?[0]);
    out.extend(&rs.process(&[vec![0.0; chunk]], None)?[0]);
    let expected = (x.len() as u64 * to as u64 / from as u64) as usize;
    Ok(out.into_iter().skip(delay).take(expected).collect())
}

#[cfg(feature = "io")]
pub fn write_wav(path: &Path, pcm: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16,
                                sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

#[cfg(feature = "io")]
/// WAV 16 bits mono en mémoire.
pub fn wav_bytes(pcm: &[f32], rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16,
                                sample_format: hound::SampleFormat::Int };
    let mut cur = std::io::Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(&mut cur, spec)?;
        for &s in pcm {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
    }
    Ok(cur.into_inner())
}

/// Enveloppe RMS par tranches de `step` secondes, normalisée à 1.
pub fn envelope(pcm: &[f32], rate: u32, step: f32, buckets: usize) -> Vec<f32> {
    let len = (rate as f32 * step) as usize;
    let mut env: Vec<f32> = (0..buckets).map(|i| {
        let a = (i * len).min(pcm.len());
        let b = ((i + 1) * len).min(pcm.len());
        if b <= a { 0.0 } else { (pcm[a..b].iter().map(|v| v * v).sum::<f32>() / (b - a) as f32).sqrt() }
    }).collect();
    let max = env.iter().cloned().fold(1e-6, f32::max);
    env.iter_mut().for_each(|v| *v /= max);
    env
}

#[cfg(all(test, feature = "io"))]
mod tests {
    use super::*;

    /// Le décodeur allégé doit donner le même son que le modèle complet.
    #[test]
    fn decodeur_seul_identique() {
        let dir = cache_dir();
        let light = dir.join("mimi-decoder-f16.safetensors");
        let body = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../echantillons/_brut/mimi_1100bps.bin");
        if !light.exists() || !dir.join(WEIGHTS_FILE).exists() || !body.exists() {
            eprintln!("poids ou échantillon absents : test ignoré");
            return;
        }
        let body = std::fs::read(body).unwrap();
        let full = Codec::load().unwrap().decode(&body).unwrap();
        let t0 = std::time::Instant::now();
        let mut dec = Decoder::load(&light).unwrap();
        eprintln!("chargement : {:.2} s", t0.elapsed().as_secs_f32());
        let t0 = std::time::Instant::now();
        let lite = dec.decode(&body).unwrap();
        eprintln!("décodeur seul : {:.2} s pour {:.1} s d'audio", t0.elapsed().as_secs_f32(), lite.len() as f32 / SAMPLE_RATE as f32);
        let t0 = std::time::Instant::now();
        let mut streamed = Vec::new();
        let mut first = None;
        dec.decode_stream(&body, 4, |p| { first.get_or_insert(t0.elapsed()); streamed.extend_from_slice(p); true }).unwrap();
        eprintln!("flux : premier son après {:.0} ms, total {:.2} s", first.unwrap().as_secs_f32() * 1000.0, t0.elapsed().as_secs_f32());
        // le décodeur se charge aussi depuis le fichier complet (secours de l'app Android)
        let from_full = Decoder::load(&dir.join(WEIGHTS_FILE)).unwrap().decode(&body).unwrap();
        assert_eq!(from_full.len(), full.len());
        let n = streamed.len().min(lite.len());
        let e2: f32 = streamed[..n].iter().zip(&lite[..n]).map(|(a, b)| (a - b).powi(2)).sum();
        let s2: f32 = lite[..n].iter().map(|a| a * a).sum();
        eprintln!("SNR flux / bloc : {:.1} dB ({} / {} échantillons)", 10.0 * (s2 / e2.max(1e-12)).log10(), streamed.len(), lite.len());
        assert_eq!(full.len(), lite.len());
        let err: f32 = full.iter().zip(&lite).map(|(a, b)| (a - b).powi(2)).sum::<f32>();
        let sig: f32 = full.iter().map(|a| a * a).sum::<f32>();
        let snr = 10.0 * (sig / err.max(1e-12)).log10();
        eprintln!("SNR float16 / float32 : {snr:.1} dB");
        assert!(snr > 30.0, "écart trop grand : {snr:.1} dB");
    }
}
