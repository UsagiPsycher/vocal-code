# vocal-code

Put a short voice message (30 s max) in a 2D code you can print on a sticker or a poster.
Scan it with your phone and it plays. No server involved: once the audio model is downloaded,
everything runs offline.

- Audio is compressed with [Mimi](https://huggingface.co/kyutai/mimi) (Kyutai), 8 codebooks, about 1.1 kbps,
  so 30 s fit in ~4.1 KB.
- The code is a black and white grid of ~245×245 modules with a finder in each corner (3 out of 4 are enough),
  small alignment markers every ~20 modules and Reed-Solomon error correction (40 % parity by default).
  It still reads with perspective, on a curved surface, slightly blurred, with glare or stains.

The apps are in French.

## Install

Download from [Releases](https://github.com/UsagiPsycher/vocal-code/releases):

- `vocal-code.exe`: Windows app, to create and read codes
- `vocal-code.apk`: Android reader (arm64, Android 10+)
- `vocalcode.exe`: command line tool

The audio model is downloaded on first launch:

| App | File | Size | Source |
|---|---|---|---|
| Windows / CLI | full Mimi model | 385 MB | Hugging Face, or the copy in Releases |
| Android | Mimi decoder only, float16 | 99 MB | Releases, or the full model on Hugging Face |

It's stored in `%LOCALAPPDATA%\vocalcode` on Windows (override with `$VOCALCODE_CACHE`) and in the app's
private storage on Android.

### Model version

Codes contain Mimi tokens, which only decode correctly with the weights that produced them. So the model
is pinned to a fixed Hugging Face revision and the file is checked against its SHA-256 after download.
If another model is ever used, it will get its own codec ID in the format and old codes will keep working.

| File | SHA-256 |
|---|---|
| `mimi-e351c8d8-checkpoint125.safetensors` | `09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50` |
| `mimi-decoder-f16.safetensors` | `5180784e951aef9fbed97e95c1120ba5c341fb3cd357ca295704063def3faf3f` |

## Usage

### Windows

- **Créer** (create): record from the mic, or open/drop an audio file. Pick a size (5, 8 or 12 cm),
  then save the PNG or print it. You can listen to what the code will contain before saving.
- **Lire** (read): drop, open or paste (Ctrl+V) a photo of a code.

### Android

Point the camera at a code. You can also open a photo from the gallery or share one to the app.
Playback starts while the rest is still decoding.

### Command line

```
vocalcode encode message.mp3 -o message.png --largeur-cm 8
vocalcode decode photo.jpg -o message.wav
vocalcode bench message.bin --essais 12
```

`encode` takes mp3, m4a, wav, flac, ogg, or a `.bin` from a previous run (it writes one next to the PNG).
The PNG has its DPI set, so printing at 100 % gives the requested width.

Options: `--ecc 0..3` (20/30/40/50 % parity), `--max-secondes` (max duration in seconds),
`--px-module` (pixels per module).

## Building

Needs stable Rust with the MSVC toolchain.

```
cargo build --release
cargo test --release
```

This produces `target\release\vocal-code.exe` and `vocalcode.exe`.

For Android you also need Android Studio (SDK + JBR), NDK r30,
`rustup target add aarch64-linux-android` and `cargo install cargo-ndk`. Then:

```
build-android.bat install
```

The APK is signed with the debug key unless `android/keystore.properties` exists.

The 99 MB decoder file in Releases is made from the full model with
(`pip install safetensors numpy`):

```
python tools/mimi_decoder_weights.py mimi-e351c8d8-checkpoint125.safetensors mimi-decoder-f16.safetensors
```

## Repository

| Folder | Contents |
|---|---|
| `crates/vocalcode-core` | code format, Reed-Solomon, rendering, detection, shared by PC and Android |
| `crates/vocalcode-audio` | Mimi encoding/decoding with the `moshi` crate, audio file I/O |
| `crates/vocalcode-cli` | `vocalcode` CLI and the robustness bench |
| `crates/vocalcode-app` | Windows app (Tauri 2, frontend in `ui/`) |
| `crates/vocalcode-android` | JNI library used by the Android app |
| `android/` | Android app (Kotlin, CameraX) |
| `tools/` | script to extract the decoder-only weights |
| `prototype-python/` | early Python prototype |

## Format (v1)

1. **Audio body**: frame count (u16 BE), then 8 × 11-bit codes per frame (12.5 frames/s).
2. **Payload**: version (1), codec (1 = Mimi, 8 codebooks), length (u16 BE), body, CRC-32 (BE).
3. **Blocks**: the payload is zero-padded to capacity and split into blocks of at most 255 bytes.
   Each block gets `round(size × ratio)` Reed-Solomon parity bytes at the end
   (GF(256), polynomial 0x11d, generator 2, fcr 0).
4. **Placement**: data modules are taken row by row in groups of 8. Byte i goes to group `perm[i]`
   (Fisher-Yates with xorshift32, seed `0x9E3779B9 ^ N`), MSB first, then XORed with an xorshift32
   mask (seed `0x6C8E9CF5`, bit 31).
5. **Format block**: 64 bits stored 4 times in 8×8 areas near the corners. `N` (10 bits), level (2 bits),
   version (4 bits), plus 6 RS bytes, XORed with `A5 3C 96 5A C3 69 0F F0`.

## Credits

- **Mimi** by Kyutai, weights under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  ([kyutai/mimi](https://huggingface.co/kyutai/mimi),
  [kyutai/moshiko-candle-bf16](https://huggingface.co/kyutai/moshiko-candle-bf16)).
  `mimi-decoder-f16.safetensors` is derived from them (decoder only, converted to float16).
- [`moshi`](https://crates.io/crates/moshi) (Kyutai) and [candle](https://github.com/huggingface/candle)
  (Hugging Face) for inference.
- **Bricolage Grotesque** font (The Bricolage Grotesque Project Authors, Atelier Triay),
  SIL Open Font License 1.1 (`crates/vocalcode-app/ui/fonts/OFL.txt`).
