package app.vocalcode.reader

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

private val PAPER = Color.parseColor("#EEF0EC")
private val INK = Color.parseColor("#1E2235")
private val PINK = Color.parseColor("#FF3EA5")
private val LINE = Color.parseColor("#D6D9D3")

/** La bande de 30 s : barres roses au fil du décodage, partie écoutée à l'encre. */
class BandView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    var message: Message? = null
    var playhead: Int = -1          // en échantillons, -1 = pas de lecture
    var onSeek: ((Int) -> Unit)? = null
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    override fun onDraw(c: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val mid = h / 2
        val slot = w / BARS
        val msg = message
        val total = msg?.let { (it.totalSamples + STEP_SAMPLES - 1) / STEP_SAMPLES } ?: 0
        val ready = msg?.let { it.available / STEP_SAMPLES } ?: 0
        val dp = resources.displayMetrics.density

        // capacité : un point toutes les 0,5 s au-delà du message, un repère toutes les 10 s
        paint.color = LINE
        var i = ((total + 4) / 5) * 5
        while (i <= BARS) {
            val x = min(w - dp, i * slot)
            val tall = i % 100 == 0 && i in 1 until BARS
            if (tall) c.drawRect(x - dp, mid - 9 * dp, x + dp, mid + 9 * dp, paint)
            else c.drawRect(x - dp, mid - 1.5f * dp, x + dp, mid + 1.5f * dp, paint)
            i += 5
        }
        // emplacement du message pas encore décodé : trait fin
        if (msg != null) {
            c.drawRect(ready * slot, mid - 0.5f * dp, total * slot, mid + 0.5f * dp, paint)
        }
        if (msg == null) return

        // barres, regroupées pour rester nettes
        val group = max(1, (3 * dp / slot).roundToInt())
        var peak = 1e-4f
        for (k in 0 until min(ready, BARS)) peak = max(peak, msg.envelope[k])
        var k = 0
        while (k < min(ready, BARS)) {
            var v = 0f
            for (j in k until min(k + group, ready)) v = max(v, msg.envelope[j])
            val x0 = (k * slot).roundToInt().toFloat()
            val x1 = (min(k + group, BARS) * slot).roundToInt().toFloat()
            val bw = max(1f, ((x1 - x0) * 0.6f).roundToInt().toFloat())
            val bh = max(2 * dp, (0.06f + 0.94f * sqrt(v / peak)) * (h - 12 * dp))
            paint.color = if (playhead >= 0 && k * STEP_SAMPLES < playhead) INK else PINK
            c.drawRect(x0, mid - bh / 2, x0 + bw, mid + bh / 2, paint)
            k += group
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        val msg = message ?: return false
        if (e.action == MotionEvent.ACTION_DOWN || e.action == MotionEvent.ACTION_MOVE) {
            val s = ((e.x / width) * MAX_SECONDS * RATE).toInt()
            onSeek?.invoke(s.coerceIn(0, msg.available))
            return true
        }
        return e.action == MotionEvent.ACTION_UP
    }
}

/** Cadre de visée : coins papier, assombrissement autour ; coins roses quand un code est lu. */
class GuideView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    var found = false
        set(v) { field = v; invalidate() }
    private val dim = Paint().apply { color = Color.argb(110, 30, 34, 53) }
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.SQUARE
    }
    private val path = Path()

    val frame: RectF
        get() {
            val side = min(width, height) * 0.84f
            val cx = width / 2f
            val cy = height * 0.44f
            return RectF(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        }

    override fun onDraw(c: Canvas) {
        val f = frame
        val dp = resources.displayMetrics.density
        c.drawRect(0f, 0f, width.toFloat(), f.top, dim)
        c.drawRect(0f, f.bottom, width.toFloat(), height.toFloat(), dim)
        c.drawRect(0f, f.top, f.left, f.bottom, dim)
        c.drawRect(f.right, f.top, width.toFloat(), f.bottom, dim)
        stroke.strokeWidth = 4 * dp
        stroke.color = if (found) PINK else PAPER
        val l = 30 * dp
        path.reset()
        path.moveTo(f.left, f.top + l); path.lineTo(f.left, f.top); path.lineTo(f.left + l, f.top)
        path.moveTo(f.right - l, f.top); path.lineTo(f.right, f.top); path.lineTo(f.right, f.top + l)
        path.moveTo(f.right, f.bottom - l); path.lineTo(f.right, f.bottom); path.lineTo(f.right - l, f.bottom)
        path.moveTo(f.left + l, f.bottom); path.lineTo(f.left, f.bottom); path.lineTo(f.left, f.bottom - l)
        c.drawPath(path, stroke)
    }
}

/** Bouton lecture/pause : cercle à l'encre. */
class PlayButton(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    var playing = false
        set(v) {
            field = v
            contentDescription = context.getString(if (v) R.string.pause else R.string.play)
            invalidate()
        }
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val tri = Path()

    init {
        isClickable = true
        isFocusable = true
    }

    override fun onDraw(c: Canvas) {
        val dp = resources.displayMetrics.density
        val r = min(width, height) / 2f
        val cx = width / 2f
        val cy = height / 2f
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2 * dp
        paint.color = INK
        c.drawCircle(cx, cy, r - dp, paint)
        paint.style = Paint.Style.FILL
        if (playing) {
            val bw = 4.5f * dp
            val bh = 16 * dp
            c.drawRect(cx - 7 * dp, cy - bh / 2, cx - 7 * dp + bw, cy + bh / 2, paint)
            c.drawRect(cx + 7 * dp - bw, cy - bh / 2, cx + 7 * dp, cy + bh / 2, paint)
        } else {
            tri.reset()
            tri.moveTo(cx - 5 * dp, cy - 9 * dp)
            tri.lineTo(cx + 9 * dp, cy)
            tri.lineTo(cx - 5 * dp, cy + 9 * dp)
            tri.close()
            c.drawPath(tri, paint)
        }
    }
}
