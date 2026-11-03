package app.vocalcode.reader

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlin.concurrent.thread
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

const val RATE = 24_000
const val MAX_SECONDS = 30
const val STEP_SAMPLES = RATE / 10     // une barre de la bande = 100 ms
const val BARS = MAX_SECONDS * 10

/** Son d'un message, rempli au fil du décodage. */
class Message(val totalSamples: Int) {
    val pcm = FloatArray(totalSamples + RATE)
    private val sums = FloatArray(BARS + 10)
    private val counts = IntArray(BARS + 10)
    val envelope = FloatArray(BARS)

    @Volatile var available = 0
        private set
    @Volatile var done = false
        private set
    @Volatile var cancelled = false

    val seconds: Float get() = totalSamples / RATE.toFloat()

    fun append(chunk: FloatArray) = synchronized(this) {
        val n = min(chunk.size, pcm.size - available)
        System.arraycopy(chunk, 0, pcm, available, n)
        for (i in 0 until n) {
            val b = (available + i) / STEP_SAMPLES
            if (b < BARS) {
                sums[b] += chunk[i] * chunk[i]
                counts[b]++
                envelope[b] = sqrt(sums[b] / counts[b])
            }
        }
        available += n
        (this as Object).notifyAll()
    }

    fun finish() = synchronized(this) {
        done = true
        (this as Object).notifyAll()
    }
}

/** Lecture en flux : joue ce qui est déjà décodé, attend la suite si besoin. */
class Player(private val msg: Message, private val onEnd: () -> Unit) {
    private val track: AudioTrack = AudioTrack.Builder()
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
        )
        .setAudioFormat(
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build()
        )
        .setTransferMode(AudioTrack.MODE_STREAM)
        .setBufferSizeInBytes(
            max(
                AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT),
                RATE / 4 * 4,
            )
        )
        .build()

    @Volatile var playing = false
        private set
    @Volatile private var startSample = 0
    @Volatile private var writePos = 0
    @Volatile private var headBase = 0
    @Volatile private var pausedAt = 0
    private var worker: Thread? = null

    /** Position de lecture, en échantillons. */
    val position: Int
        get() = if (playing) min(startSample + (track.playbackHeadPosition - headBase), writePos) else pausedAt

    fun play(from: Int = pausedAt) {
        halt()
        startSample = from.coerceIn(0, max(0, msg.available - 1))
        writePos = startSample
        headBase = track.playbackHeadPosition
        playing = true
        track.play()
        worker = thread(name = "lecture") { loop() }
    }

    fun pause() {
        if (!playing) return
        pausedAt = position
        halt()
    }

    fun toggle() = if (playing) pause() else play()

    fun seek(sample: Int) {
        if (playing) play(sample) else pausedAt = sample.coerceIn(0, msg.available)
    }

    private fun halt() {
        playing = false
        track.pause()
        track.flush()
        synchronized(msg) { (msg as Object).notifyAll() }
        worker?.join(500)
        worker = null
    }

    private fun loop() {
        while (playing) {
            val n = synchronized(msg) {
                while (playing && writePos >= msg.available && !msg.done && !msg.cancelled) (msg as Object).wait(50)
                min(msg.available - writePos, RATE / 5)
            }
            if (!playing) return
            if (n <= 0) {
                // tout est écrit : on attend la fin réelle de la lecture
                while (playing && position < writePos) Thread.sleep(15)
                if (playing) {
                    playing = false
                    pausedAt = 0
                    track.pause()
                    track.flush()
                    onEnd()
                }
                return
            }
            val w = track.write(msg.pcm, writePos, n, AudioTrack.WRITE_BLOCKING)
            if (w <= 0) return
            writePos += w
        }
    }

    fun release() {
        halt()
        track.release()
    }
}
