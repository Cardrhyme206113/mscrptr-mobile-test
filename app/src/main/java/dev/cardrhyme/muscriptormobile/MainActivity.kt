package dev.cardrhyme.muscriptormobile

import android.content.res.ColorStateList
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.view.WindowCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
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
    private lateinit var previewStatus: TextView
    private lateinit var songVolumeLabel: TextView
    private lateinit var midiVolumeLabel: TextView
    private lateinit var progress: ProgressBar
    private lateinit var downloadButton: MaterialButton
    private lateinit var pickButton: MaterialButton
    private lateinit var transcribeButton: MaterialButton
    private lateinit var cancelButton: MaterialButton
    private lateinit var saveButton: MaterialButton
    private lateinit var playPauseButton: MaterialButton
    private lateinit var restartButton: MaterialButton
    private lateinit var themeButton: MaterialButton
    private lateinit var cacheSpinner: Spinner
    private lateinit var backendSpinner: Spinner
    private lateinit var songVolume: SeekBar
    private lateinit var midiVolume: SeekBar
    private lateinit var pianoRoll: PianoRollView

    private var selectedAudio: Uri? = null
    private var selectedName: String = "audio"
    private var lastMidi: ByteArray? = null
    private var currentTask: Job? = null
    private var preview: LivePreviewController? = null
    private var noteCount = 0
    private var lastProgressPost = 0L
    private var darkMode = true

    private val pickAudio = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            runCatching {
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
            closePreview()
            selectedAudio = uri
            selectedName = displayName(uri) ?: "audio"
            audioStatus.text = selectedName
            lastMidi = null
            saveButton.isEnabled = false
            pianoRoll.reset()
            previewStatus.text = "Preview starts after a safe overlapped MIDI region is finalized"
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
        darkMode = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            .getBoolean(KEY_DARK_THEME, true)
        AppCompatDelegate.setDefaultNightMode(
            if (darkMode) AppCompatDelegate.MODE_NIGHT_YES else AppCompatDelegate.MODE_NIGHT_NO,
        )
        super.onCreate(savedInstanceState)

        WindowCompat.getInsetsController(window, window.decorView).apply {
            isAppearanceLightStatusBars = !darkMode
            isAppearanceLightNavigationBars = !darkMode
        }
        window.statusBarColor = getColor(R.color.app_background)
        window.navigationBarColor = getColor(R.color.app_background)

        downloader = ModelDownloader(this)
        buildUi()
        refreshModelState()
    }

    override fun onDestroy() {
        currentTask?.cancel()
        closePreview()
        super.onDestroy()
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16.dp, 18.dp, 16.dp, 30.dp)
            setBackgroundColor(getColor(R.color.app_background))
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            overScrollMode = View.OVER_SCROLL_NEVER
            setBackgroundColor(getColor(R.color.app_background))
            addView(root)
        }

        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val titleBlock = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        titleBlock.addView(TextView(this).apply {
            text = "MuScriptor"
            textSize = 30f
            letterSpacing = -0.025f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(getColor(R.color.app_on_surface))
        })
        titleBlock.addView(TextView(this).apply {
            text = "Offline audio to MIDI"
            textSize = 14f
            setTextColor(getColor(R.color.app_on_surface_variant))
        }, margin(top = 2))
        header.addView(titleBlock, weighted())

        themeButton = compactButton(if (darkMode) "☀  Light" else "☾  Dark").apply {
            setOnClickListener { toggleTheme() }
        }
        header.addView(themeButton, wrapWidth())
        root.addView(header, matchWidth())

        root.addView(TextView(this).apply {
            text = "INT4 transcription • 5 s windows / 4 s hop • 1 s overlap • CPU / GPU / NPU modes"
            textSize = 12.5f
            setTextColor(getColor(R.color.app_on_surface_variant))
            setLineSpacing(2f, 1f)
        }, margin(top = 12, bottom = 18, width = LinearLayout.LayoutParams.MATCH_PARENT))

        root.addView(sectionCard("Model") {
            modelStatus = statusText("Checking local model…")
            addView(modelStatus, margin(bottom = 12, width = LinearLayout.LayoutParams.MATCH_PARENT))
            downloadButton = secondaryButton("Download / verify model  ·  213 MiB").apply {
                setOnClickListener { downloadModel() }
            }
            addView(downloadButton, matchWidth())
        }, cardMargin())

        root.addView(sectionCard("Runtime") {
            addView(fieldLabel("Memory profile"))
            cacheSpinner = themedSpinner(
                listOf(
                    "Recommended · 1024 cache (~192 MiB)",
                    "Experimental · 1200 cache (~225 MiB)",
                    "Safer · 896 cache (~168 MiB)",
                    "Ultra low memory · 768 cache (~144 MiB)",
                ),
            )
            addView(cacheSpinner, fieldMargin())
            addView(hintText(
                "1200 may fit this device but uses roughly 33 MiB more KV memory than 1024. If Android kills the process, return to 1024.",
            ), margin(top = 8, width = LinearLayout.LayoutParams.MATCH_PARENT))

            addView(fieldLabel("Compute backend"), margin(top = 14))
            backendSpinner = themedSpinner(ComputeBackend.values().map { it.displayName })
            addView(backendSpinner, fieldMargin())

            addView(hintText(
                "Android chooses whether NNAPI uses the GPU or NPU. Parallel mode prepares the next overlapped window on the accelerator while the current decoder runs on CPU.",
            ), margin(top = 9, width = LinearLayout.LayoutParams.MATCH_PARENT))
        }, cardMargin())

        root.addView(sectionCard("Audio") {
            audioStatus = statusText("No audio selected")
            addView(audioStatus, margin(bottom = 12, width = LinearLayout.LayoutParams.MATCH_PARENT))

            pickButton = secondaryButton("Choose audio").apply {
                setOnClickListener { pickAudio.launch(arrayOf("audio/*")) }
            }
            addView(pickButton, matchWidth())

            transcribeButton = primaryButton("Transcribe + start live preview").apply {
                setOnClickListener { startTranscription() }
            }
            addView(transcribeButton, margin(top = 10, width = LinearLayout.LayoutParams.MATCH_PARENT))

            cancelButton = dangerButton("Cancel transcription").apply {
                visibility = View.GONE
                setOnClickListener { currentTask?.cancel() }
            }
            addView(cancelButton, margin(top = 10, width = LinearLayout.LayoutParams.MATCH_PARENT))

            progress = ProgressBar(
                this@MainActivity,
                null,
                android.R.attr.progressBarStyleHorizontal,
            ).apply {
                max = 1000
                progress = 0
                progressTintList = colorState(R.color.app_primary)
                progressBackgroundTintList = colorState(R.color.app_surface_variant)
            }
            addView(progress, margin(top = 16, width = LinearLayout.LayoutParams.MATCH_PARENT))

            inferenceStatus = hintText("Ready")
            addView(inferenceStatus, margin(top = 9, width = LinearLayout.LayoutParams.MATCH_PARENT))
        }, cardMargin())

        root.addView(sectionCard("Live preview") {
            addView(hintText(
                "Playback follows only the stable center of each overlapped window, targets a 3-second finalized lead, and slows without changing pitch when inference gets close.",
            ), margin(bottom = 10, width = LinearLayout.LayoutParams.MATCH_PARENT))

            previewStatus = statusText("Preview has not started")
            addView(previewStatus, margin(bottom = 12, width = LinearLayout.LayoutParams.MATCH_PARENT))

            val transport = LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
            }
            playPauseButton = secondaryButton("Pause").apply {
                isEnabled = false
                setOnClickListener { preview?.togglePause() }
            }
            restartButton = secondaryButton("Restart").apply {
                isEnabled = false
                setOnClickListener { preview?.restart() }
            }
            transport.addView(playPauseButton, weighted(right = 5))
            transport.addView(restartButton, weighted(left = 5))
            addView(transport, matchWidth())

            songVolumeLabel = fieldLabel("Original song  ·  $DEFAULT_SONG_PERCENT%")
            addView(songVolumeLabel, margin(top = 14))
            songVolume = volumeSeekBar(DEFAULT_SONG_PERCENT) { percent ->
                songVolumeLabel.text = "Original song  ·  $percent%"
                preview?.setSongVolume(percent / 100f)
            }
            addView(songVolume, matchWidth())

            midiVolumeLabel = fieldLabel("Generated MIDI  ·  $DEFAULT_MIDI_PERCENT%")
            addView(midiVolumeLabel, margin(top = 7))
            midiVolume = volumeSeekBar(DEFAULT_MIDI_PERCENT) { percent ->
                midiVolumeLabel.text = "Generated MIDI  ·  $percent%"
                preview?.setMidiVolume(percent / 100f)
            }
            addView(midiVolume, matchWidth())

            pianoRoll = PianoRollView(this@MainActivity).apply {
                setDarkMode(darkMode)
            }
            addView(
                pianoRoll,
                margin(
                    top = 12,
                    width = LinearLayout.LayoutParams.MATCH_PARENT,
                    height = 300.dp,
                    bottom = 12,
                ),
            )

            saveButton = secondaryButton("Export MIDI").apply {
                isEnabled = false
                setOnClickListener {
                    val stem = selectedName.substringBeforeLast('.').ifBlank { "transcription" }
                    createMidi.launch("$stem.mid")
                }
            }
            addView(saveButton, matchWidth())
        }, cardMargin())

        root.addView(TextView(this).apply {
            text = "MuScriptor model: CC BY-NC 4.0 · Use audio you have permission to process."
            textSize = 11.5f
            gravity = Gravity.CENTER
            setTextColor(getColor(R.color.app_on_surface_variant))
        }, margin(top = 16, width = LinearLayout.LayoutParams.MATCH_PARENT))

        setContentView(scroll)
    }

    private fun toggleTheme() {
        if (currentTask?.isActive == true) {
            Toast.makeText(this, "Finish or cancel the current task before changing theme", Toast.LENGTH_SHORT).show()
            return
        }
        val newDarkMode = !darkMode
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_DARK_THEME, newDarkMode)
            .apply()
        delegate.localNightMode = if (newDarkMode) {
            AppCompatDelegate.MODE_NIGHT_YES
        } else {
            AppCompatDelegate.MODE_NIGHT_NO
        }
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
                                inferenceStatus.text = "${state.fileName} · ${formatBytes(state.downloaded)} / ${formatBytes(state.total)}"
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
                inferenceStatus.text = "Download cancelled · it can resume later"
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

        closePreview()
        noteCount = 0
        lastMidi = null
        saveButton.isEnabled = false
        pianoRoll.reset()
        progress.progress = 0
        inferenceStatus.text = "Opening audio…"
        previewStatus.text = "Decoding audio before overlapped inference"
        setBusy(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val cacheLength = when (cacheSpinner.selectedItemPosition) {
            1 -> 1200
            2 -> 896
            3 -> 768
            else -> 1024
        }
        val requestedBackend = ComputeBackend.values().getOrElse(
            backendSpinner.selectedItemPosition,
        ) { ComputeBackend.CPU }

        currentTask = lifecycleScope.launch {
            try {
                val decoded = withContext(Dispatchers.Default) {
                    AudioDecoder(this@MainActivity).decode(uri) { fraction ->
                        val now = System.currentTimeMillis()
                        if (now - lastProgressPost >= 100) {
                            lastProgressPost = now
                            runOnUiThread {
                                progress.progress = (fraction * 120).toInt().coerceIn(0, 120)
                                inferenceStatus.text = "Decoding audio · ${(fraction * 100).toInt()}%"
                            }
                        }
                    }
                }

                preview = LivePreviewController(
                    context = this@MainActivity,
                    audioUri = uri,
                    onState = ::renderPreviewState,
                ).also {
                    it.setSongVolume(songVolume.progress / 100f)
                    it.setMidiVolume(midiVolume.progress / 100f)
                }
                playPauseButton.isEnabled = true

                inferenceStatus.text = String.format(
                    Locale.US,
                    "Loaded %.1f s · initializing %s · %d cache…",
                    decoded.durationSeconds,
                    requestedBackend.shortName,
                    cacheLength,
                )
                previewStatus.text = "Waiting for the first stable overlapped region"

                var activeBackendName = requestedBackend.shortName
                val result = withContext(Dispatchers.Default) {
                    MuScriptorEngine(
                        modelDir = downloader.modelDir,
                        maxCacheLength = cacheLength,
                        requestedBackend = requestedBackend,
                    ).use { engine ->
                        activeBackendName = engine.activeBackend.shortName
                        runOnUiThread {
                            inferenceStatus.text = engine.backendStatus
                            if (engine.activeBackend != requestedBackend) {
                                Toast.makeText(
                                    this@MainActivity,
                                    engine.backendStatus,
                                    Toast.LENGTH_LONG,
                                ).show()
                            }
                        }
                        engine.transcribe(
                            samples16k = decoded.samples16k,
                            onLiveEvent = { event ->
                                if (event is LiveNoteEvent.Started) noteCount += 1
                                preview?.accept(event)
                                pianoRoll.accept(event)
                            },
                            onProgress = { state ->
                                preview?.updateFinalizedFrontier(state.finalizedSeconds)
                                val now = System.currentTimeMillis()
                                if (now - lastProgressPost >= 90 || state.completedChunks == state.totalChunks) {
                                    lastProgressPost = now
                                    runOnUiThread {
                                        val windowFraction = state.completedChunks.toDouble() / state.totalChunks
                                        progress.progress = (120 + windowFraction * 880).toInt().coerceIn(0, 1000)
                                        inferenceStatus.text = buildString {
                                            append(activeBackendName)
                                            append(" · ")
                                            append(state.message)
                                            append(" · ")
                                            append(noteCount)
                                            append(" notes · ")
                                            append(state.generatedTokens)
                                            append(" tokens")
                                            if (state.lastTokenMillis > 0.0) {
                                                append(" · ")
                                                append(String.format(Locale.US, "%.1f ms/token", state.lastTokenMillis))
                                            }
                                        }
                                    }
                                }
                            },
                        )
                    }
                }

                preview?.markInferenceComplete(decoded.durationSeconds)
                lastMidi = result.midi
                progress.progress = progress.max
                saveButton.isEnabled = true
                inferenceStatus.text = "Done · $activeBackendName · ${result.notes.size} notes · ${result.generatedTokens} tokens"
                Toast.makeText(this@MainActivity, "Transcription complete", Toast.LENGTH_SHORT).show()
            } catch (_: CancellationException) {
                inferenceStatus.text = "Transcription cancelled"
                closePreview()
            } catch (error: Throwable) {
                closePreview()
                showError(error)
            } finally {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                setBusy(false)
            }
        }
    }

    private fun renderPreviewState(state: LivePreviewController.State) {
        pianoRoll.setCursor(state.positionSeconds)
        restartButton.isEnabled = state.started
        playPauseButton.isEnabled = state.started
        playPauseButton.text = if (state.playing) "Pause" else "Resume"
        previewStatus.text = when {
            !state.prepared -> "Preparing original audio player…"
            !state.started -> String.format(
                Locale.US,
                "Buffering stable overlapped MIDI · %.1f / 3.5 s",
                state.finalizedFrontierSeconds,
            )
            state.waitingForBuffer -> String.format(
                Locale.US,
                "%s · emergency buffer hold",
                formatTime(state.positionSeconds),
            )
            else -> String.format(
                Locale.US,
                "%s · lead %.1f s · %.2fx pitch-preserved",
                formatTime(state.positionSeconds),
                state.leadSeconds.coerceAtLeast(0.0),
                state.speed,
            )
        }
    }

    private fun refreshModelState() {
        modelStatus.text = if (downloader.isReady()) {
            "Ready · verified local model"
        } else {
            "Not downloaded · no account or token required"
        }
        refreshButtons()
    }

    private fun refreshButtons() {
        val busy = currentTask?.isActive == true
        downloadButton.isEnabled = !busy
        pickButton.isEnabled = !busy
        cacheSpinner.isEnabled = !busy
        backendSpinner.isEnabled = !busy
        themeButton.isEnabled = !busy
        transcribeButton.isEnabled = !busy && downloader.isReady() && selectedAudio != null
        saveButton.isEnabled = !busy && lastMidi != null
    }

    private fun setBusy(busy: Boolean) {
        cancelButton.visibility = if (busy) View.VISIBLE else View.GONE
        if (!busy) currentTask = null
        refreshButtons()
    }

    private fun closePreview() {
        preview?.close()
        preview = null
        if (::playPauseButton.isInitialized) playPauseButton.isEnabled = false
        if (::restartButton.isInitialized) restartButton.isEnabled = false
    }

    private fun showError(error: Throwable) {
        val message = error.message ?: error.javaClass.simpleName
        inferenceStatus.text = "Error · $message"
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

    private fun sectionCard(
        title: String,
        content: LinearLayout.() -> Unit,
    ): MaterialCardView {
        val card = MaterialCardView(this).apply {
            radius = 3.dp.toFloat()
            cardElevation = 0f
            strokeWidth = 1.dp
            setStrokeColor(getColor(R.color.app_outline))
            setCardBackgroundColor(getColor(R.color.app_surface))
        }
        val body = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16.dp, 16.dp, 16.dp, 16.dp)
            addView(TextView(this@MainActivity).apply {
                text = title
                textSize = 17f
                setTypeface(typeface, Typeface.BOLD)
                setTextColor(getColor(R.color.app_on_surface))
            }, margin(bottom = 13))
            content()
        }
        card.addView(body)
        return card
    }

    private fun statusText(value: String) = TextView(this).apply {
        text = value
        textSize = 14f
        setTextColor(getColor(R.color.app_on_surface))
        setLineSpacing(2f, 1f)
    }

    private fun fieldLabel(value: String) = TextView(this).apply {
        text = value
        textSize = 12.5f
        setTypeface(typeface, Typeface.BOLD)
        setTextColor(getColor(R.color.app_on_surface_variant))
    }

    private fun hintText(value: String) = TextView(this).apply {
        text = value
        textSize = 12.5f
        setTextColor(getColor(R.color.app_on_surface_variant))
        setLineSpacing(3f, 1f)
    }

    private fun primaryButton(value: String) = baseButton(value).apply {
        backgroundTintList = colorState(R.color.app_primary)
        setTextColor(getColor(R.color.app_on_primary))
        strokeWidth = 0
    }

    private fun secondaryButton(value: String) = baseButton(value).apply {
        backgroundTintList = colorState(R.color.app_surface_variant)
        setTextColor(getColor(R.color.app_on_surface))
        strokeWidth = 1.dp
        strokeColor = colorState(R.color.app_outline)
    }

    private fun dangerButton(value: String) = baseButton(value).apply {
        backgroundTintList = colorState(R.color.app_error)
        setTextColor(getColor(R.color.app_on_error))
        strokeWidth = 0
    }

    private fun compactButton(value: String) = secondaryButton(value).apply {
        minHeight = 42.dp
        minimumHeight = 42.dp
        setPadding(14.dp, 0, 14.dp, 0)
        textSize = 12.5f
    }

    private fun baseButton(value: String) = MaterialButton(this).apply {
        text = value
        isAllCaps = false
        textSize = 14f
        cornerRadius = 3.dp
        minHeight = 50.dp
        minimumHeight = 50.dp
        insetTop = 0
        insetBottom = 0
        setTypeface(typeface, Typeface.BOLD)
    }

    private fun themedSpinner(items: List<String>) = Spinner(this).apply {
        adapter = object : ArrayAdapter<String>(
            this@MainActivity,
            android.R.layout.simple_spinner_item,
            items,
        ) {
            init {
                setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }

            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View =
                styleSpinnerText(super.getView(position, convertView, parent) as TextView, false)

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View =
                styleSpinnerText(super.getDropDownView(position, convertView, parent) as TextView, true)
        }
        minimumHeight = 50.dp
        setPadding(10.dp, 0, 10.dp, 0)
        background = sharpFieldBackground()
        popupBackgroundDrawable?.setTint(getColor(R.color.app_surface))
    }

    private fun sharpFieldBackground() = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = 2.dp.toFloat()
        setColor(getColor(R.color.app_surface_variant))
        setStroke(1.dp, getColor(R.color.app_outline))
    }

    private fun styleSpinnerText(view: TextView, dropdown: Boolean): TextView = view.apply {
        textSize = 14f
        setTextColor(getColor(R.color.app_on_surface))
        setPadding(12.dp, if (dropdown) 14.dp else 0, 12.dp, if (dropdown) 14.dp else 0)
        if (dropdown) setBackgroundColor(getColor(R.color.app_surface))
    }

    private fun volumeSeekBar(initial: Int, onChanged: (Int) -> Unit) = SeekBar(this).apply {
        max = 100
        progress = initial
        progressTintList = colorState(R.color.app_primary)
        thumbTintList = colorState(R.color.app_primary)
        progressBackgroundTintList = colorState(R.color.app_surface_variant)
        setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, value: Int, fromUser: Boolean) {
                onChanged(value)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })
    }

    private fun colorState(colorRes: Int) = ColorStateList.valueOf(getColor(colorRes))

    private fun matchWidth() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    )

    private fun wrapWidth() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.WRAP_CONTENT,
        LinearLayout.LayoutParams.WRAP_CONTENT,
    )

    private fun weighted(left: Int = 0, right: Int = 0) = LinearLayout.LayoutParams(
        0,
        LinearLayout.LayoutParams.WRAP_CONTENT,
        1f,
    ).apply {
        setMargins(left.dp, 0, right.dp, 0)
    }

    private fun cardMargin() = margin(
        top = 10,
        width = LinearLayout.LayoutParams.MATCH_PARENT,
    )

    private fun fieldMargin() = margin(
        top = 6,
        width = LinearLayout.LayoutParams.MATCH_PARENT,
    )

    private fun margin(
        top: Int = 0,
        bottom: Int = 0,
        width: Int = LinearLayout.LayoutParams.WRAP_CONTENT,
        height: Int = LinearLayout.LayoutParams.WRAP_CONTENT,
    ) = LinearLayout.LayoutParams(width, height).apply {
        setMargins(0, top.dp, 0, bottom.dp)
    }

    private val Int.dp: Int
        get() = (this * resources.displayMetrics.density).toInt()

    private fun formatBytes(bytes: Long): String = String.format(
        Locale.US,
        "%.1f MiB",
        bytes / 1024.0 / 1024.0,
    )

    private fun formatTime(seconds: Double): String {
        val total = seconds.toInt().coerceAtLeast(0)
        return "%d:%02d".format(Locale.US, total / 60, total % 60)
    }

    companion object {
        private const val PREFS_NAME = "muscriptor_ui"
        private const val KEY_DARK_THEME = "dark_theme"
        private const val DEFAULT_SONG_PERCENT = 20
        private const val DEFAULT_MIDI_PERCENT = 80
    }
}
