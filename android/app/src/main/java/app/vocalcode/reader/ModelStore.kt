package app.vocalcode.reader

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Poids du décodeur Mimi (Kyutai, CC-BY 4.0), téléchargés au premier lancement puis gardés.
 * Version figée : les codes imprimés dépendent de ces poids exacts.
 */
object ModelStore {
    private class Source(val url: String, val file: String, val sha256: String, val size: Long)

    private val SOURCES = listOf(
        // décodeur seul en float16 (tools/mimi_decoder_weights.py), publié avec le projet
        Source(
            "https://github.com/UsagiPsycher/vocal-code/releases/download/modele-mimi-v1/mimi-decoder-f16.safetensors",
            "mimi-decoder-f16.safetensors",
            "5180784e951aef9fbed97e95c1120ba5c341fb3cd357ca295704063def3faf3f",
            98_790_754,
        ),
        // secours : modèle complet chez Kyutai, à une révision figée (le décodeur n'en lit que sa part)
        Source(
            "https://huggingface.co/kyutai/moshiko-candle-bf16/resolve/50c436c9849a20685a47f087dc243f6bddac207b/tokenizer-e351c8d8-checkpoint125.safetensors",
            "mimi-e351c8d8-checkpoint125.safetensors",
            "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
            384_644_900,
        ),
    )

    /** Renvoie le fichier de poids, en le téléchargeant si besoin ; null si aucune source n'a marché. */
    fun ensure(dir: File, onProgress: (Int) -> Unit): File? {
        SOURCES.map { File(dir, it.file) }.firstOrNull { f -> SOURCES.any { it.file == f.name && f.length() == it.size } }
            ?.let { return it }
        for (src in SOURCES) {
            val dst = File(dir, src.file)
            val tmp = File(dir, src.file + ".part")
            try {
                download(src, tmp, onProgress)
                if (tmp.renameTo(dst)) return dst
            } catch (e: Exception) {
                tmp.delete()
            }
        }
        return null
    }

    private fun download(src: Source, tmp: File, onProgress: (Int) -> Unit) {
        val conn = URL(src.url).openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 30_000
        conn.instanceFollowRedirects = true
        if (conn.responseCode != 200) throw IllegalStateException("HTTP ${conn.responseCode}")
        val md = MessageDigest.getInstance("SHA-256")
        var done = 0L
        var last = -1
        conn.inputStream.use { input ->
            tmp.outputStream().buffered(1 shl 20).use { out ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                    out.write(buf, 0, n)
                    done += n
                    val pct = (done * 100 / src.size).toInt().coerceAtMost(100)
                    if (pct != last) { last = pct; onProgress(pct) }
                }
            }
        }
        val hex = md.digest().joinToString("") { "%02x".format(it) }
        if (hex != src.sha256) throw IllegalStateException("empreinte inattendue")
    }
}
