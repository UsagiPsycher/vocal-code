package app.vocalcode.reader

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Pont vers la bibliothèque Rust (crate vocalcode-android). */
object Native {
    init {
        System.loadLibrary("vocalcode")
    }

    /** Charge le décodeur Mimi allégé. */
    @JvmStatic external fun load(weightsPath: String): Boolean

    /** Cherche et lit un code dans une image en niveaux de gris ; null si rien de lisible. */
    @JvmStatic external fun scan(gray: ByteArray, width: Int, height: Int, rowStride: Int): ByteArray?

    /** Décode le son en flux ; `sink` reçoit chaque morceau et renvoie false pour arrêter. */
    @JvmStatic external fun decode(body: ByteArray, sink: PcmSink): Boolean
}

fun interface PcmSink {
    fun onPcm(pcm: FloatArray): Boolean
}

/** Résultat d'un scan : marge de correction utilisée, contour du code dans l'image, corps audio. */
class ScanResult(val usage: Float, val outline: FloatArray, val body: ByteArray) {
    /** Durée annoncée par l'en-tête du corps (trames Mimi à 12,5 par seconde). */
    val seconds: Float
        get() = (((body[0].toInt() and 0xff) shl 8) or (body[1].toInt() and 0xff)) / 12.5f

    companion object {
        fun parse(raw: ByteArray): ScanResult {
            val b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
            val usage = b.float
            val outline = FloatArray(8) { b.float }
            return ScanResult(usage, outline, raw.copyOfRange(36, raw.size))
        }
    }
}
