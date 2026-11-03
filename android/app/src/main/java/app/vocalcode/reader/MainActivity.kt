package app.vocalcode.reader

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.util.Log
import android.util.Size
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.animation.PathInterpolator
import android.widget.ImageButton
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import kotlin.concurrent.thread
import kotlin.math.max
import kotlin.math.min

private const val TAG = "vocalcode"

class MainActivity : ComponentActivity() {

    private lateinit var preview: PreviewView
    private lateinit var guide: GuideView
    private lateinit var status: TextView
    private lateinit var sheet: View
    private lateinit var clock: TextView
    private lateinit var band: BandView
    private lateinit var play: PlayButton
    private lateinit var note: TextView
    private lateinit var torch: ImageButton

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var camera: Camera? = null
    private var torchOn = false

    @Volatile private var scanning = true
    private var message: Message? = null
    private var player: Player? = null

    @Volatile private var modelReady = CountDownLatch(1)
    @Volatile private var modelOk = false
    @Volatile private var modelProgress = -1   // en %, -1 = pas de téléchargement en cours

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else showStatus(getString(R.string.no_camera), sticky = true)
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) scanUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        guide = findViewById(R.id.guide)
        status = findViewById(R.id.status)
        sheet = findViewById(R.id.sheet)
        clock = findViewById(R.id.clock)
        band = findViewById(R.id.band)
        play = findViewById(R.id.play)
        note = findViewById(R.id.note)
        torch = findViewById(R.id.torch)

        clock.fontVariationSettings = "'wdth' 75, 'wght' 600, 'opsz' 96"
        clock.fontFeatureSettings = "tnum"

        // marges système (bord à bord)
        val bar = findViewById<View>(R.id.bar)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val s = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val dp = resources.displayMetrics.density
            status.updatePadding(top = s.top + (20 * dp).toInt())
            bar.updatePadding(bottom = s.bottom + (28 * dp).toInt())
            sheet.updatePadding(bottom = s.bottom + (28 * dp).toInt())
            insets
        }

        // modèle indisponible : toucher le message pour réessayer
        status.setOnClickListener { if (!modelOk && modelReady.count == 0L) { showStatus("…", sticky = true); prepareModel() } }
        findViewById<View>(R.id.gallery).setOnClickListener {
            pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        torch.setOnClickListener {
            torchOn = !torchOn
            camera?.cameraControl?.enableTorch(torchOn)
            torch.alpha = if (torchOn) 1f else 0.7f
        }
        torch.alpha = 0.7f
        play.setOnClickListener { player?.toggle(); refresh() }
        findViewById<View>(R.id.again).setOnClickListener { closeMessage() }
        band.onSeek = { s -> player?.seek(s); refresh() }

        preview.setOnTouchListener { v, e ->
            if (e.action == MotionEvent.ACTION_UP) {
                val p = preview.meteringPointFactory.createPoint(e.x, e.y)
                camera?.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(p).build())
                v.performClick()
            }
            true
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (sheet.visibility == View.VISIBLE) closeMessage() else finish()
            }
        })

        prepareModel()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askCamera.launch(Manifest.permission.CAMERA)
        }
        handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val uri: Uri? = when (intent?.action) {
            Intent.ACTION_SEND -> intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            Intent.ACTION_VIEW -> intent.data
            else -> null
        }
        if (uri != null) scanUri(uri)
    }

    // ---------- modèle audio ----------

    private fun prepareModel() {
        val latch = CountDownLatch(1)
        modelReady = latch
        thread(name = "modèle") {
            val file = ModelStore.ensure(filesDir) { pct ->
                modelProgress = pct
                runOnUiThread { if (scanning) showStatus(getString(R.string.downloading, pct), sticky = true) }
            }
            modelProgress = -1
            modelOk = file != null && Native.load(file.absolutePath)
            latch.countDown()
            runOnUiThread {
                if (modelOk) {
                    if (status.text != getString(R.string.aim)) showStatus(getString(R.string.aim), sticky = true)
                } else {
                    showStatus(getString(R.string.model_error), sticky = true)
                    note.setText(R.string.model_error)
                }
            }
        }
    }

    // ---------- caméra ----------

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val ratio = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
            val prev = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(ratio).build())
                .build()
                .also { it.surfaceProvider = preview.surfaceProvider }
            // le code a besoin de beaucoup de pixels : on demande la plus haute résolution d'analyse possible
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(ratio)
                        .setResolutionStrategy(
                            ResolutionStrategy(Size(4000, 3000), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                        )
                        .build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
            val analyzer = ImageAnalysis.Analyzer { img ->
                if (!scanning) { img.close(); return@Analyzer }
                val plane = img.planes[0]
                val buf = plane.buffer
                val bytes = ByteArray(buf.remaining())
                buf.get(bytes)
                val w = img.width
                val h = img.height
                val stride = plane.rowStride
                img.close()
                val t = SystemClock.elapsedRealtime()
                val raw = Native.scan(bytes, w, h, stride)
                Log.i(TAG, "scan ${w}x$h : ${SystemClock.elapsedRealtime() - t} ms, ${if (raw != null) "code lu" else "rien"}")
                if (raw == null) return@Analyzer
                val res = ScanResult.parse(raw)
                runOnUiThread { if (scanning) onFound(res) }
            }
            analysis.setAnalyzer(analysisExecutor, analyzer)
            provider.unbindAll()
            camera = try {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, prev, analysis)
            } catch (e: Exception) {
                // certaines caméras refusent la haute résolution d'analyse : on retombe sur la valeur par défaut
                val basic = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                    .also { a -> a.setAnalyzer(analysisExecutor, analyzer) }
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, prev, basic)
            }
            camera?.cameraControl?.enableTorch(torchOn)
        }, ContextCompat.getMainExecutor(this))
    }

    // ---------- photo importée ----------

    private fun scanUri(uri: Uri) {
        showStatus("…", sticky = true)
        thread(name = "photo") {
            val res = try {
                val src = ImageDecoder.createSource(contentResolver, uri)
                val bmp = ImageDecoder.decodeBitmap(src) { dec, info, _ ->
                    dec.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    val m = max(info.size.width, info.size.height)
                    if (m > 4000) {
                        val k = 4000f / m
                        dec.setTargetSize((info.size.width * k).toInt(), (info.size.height * k).toInt())
                    }
                }
                Native.scan(toGray(bmp), bmp.width, bmp.height, bmp.width)?.let { ScanResult.parse(it) }
            } catch (e: Exception) {
                null
            }
            runOnUiThread {
                if (res != null) onFound(res)
                else showStatus(getString(R.string.no_code))
            }
        }
    }

    private fun toGray(bmp: Bitmap): ByteArray {
        val w = bmp.width
        val h = bmp.height
        val row = IntArray(w)
        val out = ByteArray(w * h)
        for (y in 0 until h) {
            bmp.getPixels(row, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val c = row[x]
                val l = (77 * ((c shr 16) and 0xff) + 150 * ((c shr 8) and 0xff) + 29 * (c and 0xff)) shr 8
                out[y * w + x] = l.toByte()
            }
        }
        return out
    }

    // ---------- message ----------

    private fun onFound(res: ScanResult) {
        scanning = false
        guide.found = true
        guide.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        showStatus(getString(R.string.aim), sticky = true)

        val total = max(1, (res.seconds * RATE).toInt())
        val msg = Message(total)
        message = msg
        player?.release()
        val p = Player(msg) { runOnUiThread { refresh() } }
        player = p
        band.message = msg
        band.playhead = -1
        note.visibility = if (res.usage > 0.75f) View.VISIBLE else View.GONE
        note.setText(R.string.barely)
        openSheet()

        val t0 = SystemClock.elapsedRealtime()
        var started = false
        val ready = modelReady
        thread(name = "mimi") {
            ready.await()
            if (!modelOk) { msg.finish(); return@thread }
            Native.decode(res.body) { chunk ->
                if (msg.cancelled) return@decode false
                msg.append(chunk)
                if (!started) {
                    // on démarre dès que le décodage peut tenir le rythme jusqu'au bout sans coupure
                    val decoded = msg.available / RATE.toFloat()
                    val speed = decoded / max(1e-3f, (SystemClock.elapsedRealtime() - t0) / 1000f)
                    val need = msg.seconds
                    if (decoded >= min(0.6f, need) && (speed >= 1.15f || decoded >= need * (1 - speed / 1.15f))) {
                        started = true
                        Log.i(TAG, "lecture lancée après ${SystemClock.elapsedRealtime() - t0} ms (vitesse x$speed)")
                        runOnUiThread { if (message === msg) p.play(0) }
                    }
                }
                true
            }
            msg.finish()
            val wall = (SystemClock.elapsedRealtime() - t0) / 1000f
            Log.i(TAG, "décodage : ${msg.available / RATE.toFloat()} s d'audio en $wall s (x${msg.available / RATE.toFloat() / wall})")
            if (!started && !msg.cancelled) runOnUiThread { if (message === msg) p.play(0) }
        }
        tick()
    }

    private fun openSheet() {
        sheet.visibility = View.VISIBLE
        sheet.post {
            sheet.translationY = sheet.height.toFloat()
            sheet.animate().translationY(0f).setDuration(320)
                .setInterpolator(PathInterpolator(0.2f, 0.7f, 0.2f, 1f)).start()
        }
        findViewById<View>(R.id.bar).visibility = View.INVISIBLE
    }

    private fun closeMessage() {
        message?.cancelled = true
        message = null
        player?.release()
        player = null
        sheet.animate().translationY(sheet.height.toFloat()).setDuration(220).withEndAction {
            sheet.visibility = View.GONE
        }.start()
        findViewById<View>(R.id.bar).visibility = View.VISIBLE
        guide.found = false
        scanning = true
    }

    /** Rafraîchit l'affichage tant que le message est ouvert. */
    private fun tick() {
        if (message == null) return
        refresh()
        clock.postOnAnimation { tick() }
    }

    private fun refresh() {
        val msg = message ?: return
        val p = player
        val pos = p?.position ?: 0
        val isPlaying = p?.playing == true
        band.playhead = if (isPlaying || pos > 0) pos else -1
        band.invalidate()
        play.playing = isPlaying
        val shown = if (isPlaying || pos > 0) pos / RATE.toFloat() else 0f
        setClock(shown, msg.seconds)
        val pct = modelProgress
        if (pct >= 0) {
            note.text = getString(R.string.preparing_sound, pct)
            note.visibility = View.VISIBLE
        } else if (!modelOk && modelReady.count == 0L) {
            note.setText(R.string.model_error)
            note.visibility = View.VISIBLE
        }
    }

    private fun setClock(now: Float, total: Float) {
        val a = fmt(now)
        val b = "/" + fmt(total)
        val s = SpannableStringBuilder(a + b)
        s.setSpan(RelativeSizeSpan(0.42f), a.length, s.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        s.setSpan(ForegroundColorSpan(getColor(R.color.mute)), a.length, s.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        clock.text = s
    }

    private fun fmt(s: Float): String {
        val t = s.toInt()
        return "%d:%02d".format(t / 60, t % 60)
    }

    // ---------- statut ----------

    private val resetStatus = Runnable { status.text = getString(R.string.aim) }

    private fun showStatus(text: String, sticky: Boolean = false) {
        status.removeCallbacks(resetStatus)
        status.text = text
        if (!sticky) status.postDelayed(resetStatus, 3500)
    }

    override fun onPause() {
        super.onPause()
        player?.pause()
    }

    override fun onDestroy() {
        message?.cancelled = true
        player?.release()
        analysisExecutor.shutdown()
        super.onDestroy()
    }
}
