package dev.cardrhyme.muscriptormobile

import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private lateinit var downloader: ModelDownloader
    private lateinit var modelStatus: TextView
    private lateinit var audioStatus: TextView
    private lateinit var inferenceStatus: TextView
    private lateinit var progress: ProgressBar
    private lateinit var downloadButton: Button
    private lateinit var pickButton: Button
    private lateinit var transcribeButton: Button
    private lateinit var cancelButton: Button
    private lateinit var saveButton: Button
    private lateinit var cacheSpinner: Spinner
    private lateinit var pianoRoll: PianoRollView

    private var selectedAudio: Uri? = null
    private var selectedName: String = "audio"
    private var lastMidi: ByteArray? = null
    private var currentTask: Job? = null
    private var noteCount = 0
    private var lastProgressPost = 0L

    private val pickAudio = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            runCatching {
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
            selectedAudio = uri
            selectedName = displayName(uri) ?: "audio"
            audioStatus.text = selectedName
            lastMidi = null
            saveButton.isEnabled = false
            pianoRoll.reset()
            refreshButtons()
        }
    }

    private val createMidi = registerForActivityResult(
        ActivityResultContracts.CreateDocument("audio/midi"),
    ) { uri ->
        val bytes = lastMidi
        if (uri != null && bytes != null) {
            runCatching {
                contentResolver.openOutputStream(uri, "w")!!.use { it.write(bytes) }
            }.onSuccess {
                Toast.makeText(this, "MIDI saved", Toast.LENGTH_SHORT).show()
            }.onFailure(::showError)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        downloader = ModelDownloader(this)
        buildUi()
        refreshModelState()
    }

    override fun onDestroy() {
        currentTask?.cancel()
        super.onDestroy()
    }

    private fun buildUi() {
        val density = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(
                (18 * density).toInt(),
                (18 * density).toInt(),
                (18 * density).toInt(),
                (28 * density).toInt(),
            )
        }
        val scroll = ScrollView(this).apply { addView(root) }

        root.addView(TextView(this).apply {
            text = "MuScriptor Mobile"
            textSize = 28f
            setTypeface(typeface, Typeface.BOLD)
        })
        root.addView(TextView(this).apply {
            text = "Offline INT4 audio-to-MIDI • live token streaming"
            textSize = 14f
            alpha = 0.75f
        }, margin(bottom = 18))

        root.addView(sectionTitle("Model"))
        modelStatus = TextView(this)
        root.addView(modelStatus, margin(bottom = 8))
        downloadButton = Button(this).apply {
            text = "Download / verify model (213 MiB)"
            setOnClickListener { downloadModel() }
        }
        root.addView(downloadButton, matchWidth())

        root.addView(sectionTitle("Memory profile"), margin(top = 18))
        cacheSpinner = Spinner(this)
        val profiles = listOf(
            "Balanced • 1536 cache (~288 MiB)",
            "Full quality • 2504 cache (~470 MiB)",
            "Low memory • 1024 cache (~192 MiB)",
        )
        cacheSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            profiles,
        )
        root.addView(cacheSpinner, matchWidth())

        root.addView(sectionTitle("Audio"), margin(top = 18))
        audioStatus = TextView(this).apply { text = "No audio selected" }
        root.addView(audioStatus, margin(bottom = 8))
        pickButton = Button(this).apply {
            text = "Choose audio"
            setOnClickListener { pickAudio.launch(arrayOf("audio/*")) }
        }
        root.addView(pickButton, matchWidth())

        transcribeButton = Button(this).apply {
            text = "Transcribe locally"
            setOnClickListener { startTranscription() }
        }
        root.addView(
            transcribeButton,
            margin(top = 10, width = LinearLayout.LayoutParams.MATCH_PARENT),
        )

        cancelButton = Button(this).apply {
            text = "Cancel"
            visibility = View.GONE
            setOnClickListener { currentTask?.cancel() }
        }
        root.addView(
            cancelButton,
            margin(top = 8, width = LinearLayout.LayoutParams.MATCH_PARENT),
        )

        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 1000
            progress = 0
        }
        root.addView(
            progress,
            margin(top = 18, width = LinearLayout.LayoutParams.MATCH_PARENT),
        )
        inferenceStatus = TextView(this).apply {
            text = "Ready"
            gravity = Gravity.START
        }
        root.addView(inferenceStatus, margin(top = 8, bottom = 10))

        pianoRoll = PianoRollView(this)
        root.addView(
            pianoRoll,
            margin(
                width = LinearLayout.LayoutParams.MATCH_PARENT,
                height = (320 * density).toInt(),
                bottom = 12,
            ),
        )

        saveButton = Button(this).apply {
            text = "Export MIDI"
            isEnabled = false
            setOnClickListener {
                val stem = selectedName.substringBeforeLast('.').ifBlank { "transcription" }
                createMidi.launch("$stem.mid")
            }
        }
        root.addView(saveButton, matchWidth())

        root.addView(TextView(this).apply {
            text = "The model is CC BY-NC 4.0. Only transcribe audio you have the necessary rights to use."
            textSize = 12f
            alpha = 0.65f
        }, margin(top = 18))

        setContentView(scroll)
    }

    private fun downloadModel() {
        if (currentTask?.isActive == true) return
        setBusy(true)
        progress.progress = 0
        currentTask = lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    downloader.download(
                        onProgress = { state ->
                            runOnUiThread {
                                progress.progress = (
                                    state.downloaded.toDouble() / state.total * progress.max
                                ).toInt().coerceIn(0, progress.max)
                                inferenceStatus.text = "${state.fileName} • ${formatBytes(state.downloaded)} / ${formatBytes(state.total)}"
                            }
                        },
                        onStatus = { status ->
                            runOnUiThread { modelStatus.text = status }
                        },
                    )
                }
                progress.progress = progress.max
                Toast.makeText(this@MainActivity, "Model ready", Toast.LENGTH_SHORT).show()
            } catch (_: CancellationException) {
                inferenceStatus.text = "Download cancelled; it can resume later"
            } catch (error: Throwable) {
                showError(error)
            } finally {
                setBusy(false)
                refreshModelState()
            }
        }
    }

    private fun startTranscription() {
        val uri = selectedAudio ?: return
        if (!downloader.isReady() || currentTask?.isActive == true) return

        noteCount = 0
        lastMidi = null
        saveButton.isEnabled = false
        pianoRoll.reset()
        progress.progress = 0
        inferenceStatus.text = "Opening audio…"
        setBusy(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val cacheLength = when (cacheSpinner.selectedItemPosition) {
            1 -> 2504
            2 -> 1024
            else -> 1536
        }

        currentTask = lifecycleScope.launch {
            try {
                val decoded = withContext(Dispatchers.Default) {
                    AudioDecoder(this@MainActivity).decode(uri) { fraction ->
                        val now = System.currentTimeMillis()
                        if (now - lastProgressPost >= 100) {
                            lastProgressPost = now
                            runOnUiThread {
                                progress.progress = (fraction * 120).toInt().coerceIn(0, 120)
                                inferenceStatus.text = "Decoding audio • ${(fraction * 100).toInt()}%"
                            }
                        }
                    }
                }

                inferenceStatus.text = String.format(
                    Locale.US,
                    "Loaded %.1f s • initializing model…",
                    decoded.durationSeconds,
                )

                val result = withContext(Dispatchers.Default) {
                    MuScriptorEngine(
                        modelDir = downloader.modelDir,
                        maxCacheLength = cacheLength,
                    ).use { engine ->
                        engine.transcribe(
                            samples16k = decoded.samples16k,
                            onLiveEvent = { event ->
                                if (event is LiveNoteEvent.Started) noteCount += 1
                                pianoRoll.post { pianoRoll.accept(event) }
                            },
                            onProgress = { state ->
                                val now = System.currentTimeMillis()
                                if (now - lastProgressPost >= 70 || state.completedChunks == state.totalChunks) {
                                    lastProgressPost = now
                                    runOnUiThread {
                                        val chunkFraction = state.completedChunks.toDouble() / state.totalChunks
                                        progress.progress = (120 + chunkFraction * 880).toInt().coerceIn(0, 1000)
                                        pianoRoll.setCursor(state.completedChunks * 5.0)
                                        inferenceStatus.text = buildString {
                                            append(state.message)
                                            append(" • ")
                                            append(noteCount)
                                            append(" notes • ")
                                            append(state.generatedTokens)
                                            append(" tokens")
                                            if (state.lastTokenMillis > 0.0) {
                                                append(" • ")
                                                append(String.format(Locale.US, "%.1f ms/token", state.lastTokenMillis))
                                            }
                                        }
                                    }
                                }
                            },
                        )
                    }
                }

                lastMidi = result.midi
                progress.progress = progress.max
                saveButton.isEnabled = true
                inferenceStatus.text = "Done • ${result.notes.size} notes • ${result.generatedTokens} tokens"
                Toast.makeText(this@MainActivity, "Transcription complete", Toast.LENGTH_SHORT).show()
            } catch (_: CancellationException) {
                inferenceStatus.text = "Transcription cancelled"
            } catch (error: Throwable) {
                showError(error)
            } finally {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                setBusy(false)
            }
        }
    }

    private fun refreshModelState() {
        modelStatus.text = if (downloader.isReady()) {
            "Ready • verified local model"
        } else {
            "Not downloaded • no Hugging Face account/token required"
        }
        refreshButtons()
    }

    private fun refreshButtons() {
        val busy = currentTask?.isActive == true
        downloadButton.isEnabled = !busy
        pickButton.isEnabled = !busy
        cacheSpinner.isEnabled = !busy
        transcribeButton.isEnabled = !busy && downloader.isReady() && selectedAudio != null
        saveButton.isEnabled = !busy && lastMidi != null
    }

    private fun setBusy(busy: Boolean) {
        cancelButton.visibility = if (busy) View.VISIBLE else View.GONE
        if (!busy) currentTask = null
        refreshButtons()
    }

    private fun showError(error: Throwable) {
        val message = error.message ?: error.javaClass.simpleName
        inferenceStatus.text = "Error: $message"
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private fun displayName(uri: Uri): String? = contentResolver.query(
        uri,
        arrayOf(OpenableColumns.DISPLAY_NAME),
        null,
        null,
        null,
    )?.use { cursor ->
        if (cursor.moveToFirst()) cursor.getString(0) else null
    }

    private fun sectionTitle(text: String) = TextView(this).apply {
        this.text = text
        textSize = 18f
        setTypeface(typeface, Typeface.BOLD)
    }

    private fun matchWidth() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    )

    private fun margin(
        top: Int = 0,
        bottom: Int = 0,
        width: Int = LinearLayout.LayoutParams.WRAP_CONTENT,
        height: Int = LinearLayout.LayoutParams.WRAP_CONTENT,
    ) = LinearLayout.LayoutParams(width, height).apply {
        val density = resources.displayMetrics.density
        setMargins(0, (top * density).toInt(), 0, (bottom * density).toInt())
    }

    private fun formatBytes(bytes: Long): String = String.format(
        Locale.US,
        "%.1f MiB",
        bytes / 1024.0 / 1024.0,
    )
}
