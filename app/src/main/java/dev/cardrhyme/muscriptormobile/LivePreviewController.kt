package dev.cardrhyme.muscriptormobile

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.webkit.MimeTypeMap
import androidx.media3.common.AudioAttributes
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.util.PriorityQueue
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.abs

/**
 * Uses the original recording as the master clock and schedules finalized MIDI notes against it.
 *
 * Document-provider URIs are first copied to a private local file on a background thread. This
 * gives ExoPlayer a stable, seekable source with a known MIME type while inference is saturating the
 * device, instead of leaving the player indefinitely buffering on a slow/opaque content provider.
 */
class LivePreviewController(
    context: Context,
    audioUri: Uri,
    private val onState: (State) -> Unit,
) : Closeable {
    data class State(
        val prepared: Boolean,
        val started: Boolean,
        val playing: Boolean,
        val waitingForBuffer: Boolean,
        val positionSeconds: Double,
        val finalizedFrontierSeconds: Double,
        val leadSeconds: Double,
        val speed: Float,
    )

    private sealed interface Action {
        data class On(
            val id: Long,
            val program: Int,
            val pitch: Int,
            val isDrum: Boolean,
            val offsetSeconds: Double,
        ) : Action

        data class Off(val id: Long) : Action
    }

    private data class ScheduledEvent(
        val timeSeconds: Double,
        val priority: Int,
        val sequence: Long,
        val action: Action,
    )

    private val eventComparator = compareBy<ScheduledEvent> { it.timeSeconds }
        .thenBy { it.priority }
        .thenBy { it.sequence }

    private val appContext = context.applicationContext
    private val originalAudioUri = audioUri
    private val sourceMimeType = appContext.contentResolver.getType(audioUri)
    private val mainHandler = Handler(Looper.getMainLooper())
    private val sourceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "muscriptor-preview-source").apply { isDaemon = true }
    }
    private var sourceFuture: Future<*>? = null
    private var stagedSourceFile: File? = null

    private val eventLock = Any()
    private val allEvents = ArrayList<ScheduledEvent>()
    private val pendingEvents = PriorityQueue(eventComparator)
    private val finalizedIds = HashSet<Long>()
    private var eventSequence = 0L

    private val synth = LiveMidiSynth()
    private var player: ExoPlayer? = null
    private var requestedSongVolume = DEFAULT_SONG_VOLUME

    @Volatile
    private var finalizedFrontierSeconds = 0.0

    @Volatile
    private var inferenceComplete = false

    @Volatile
    private var audioDurationSeconds = 0.0

    @Volatile
    private var lastKnownPositionSeconds = 0.0

    private var closed = false
    private var started = false
    private var userPaused = false
    private var emergencyHold = false
    private var currentSpeed = 1f
    private var lastUiUpdateMs = 0L
    private var lastSpeedApplyMs = 0L
    private var lastUiSignature = -1
    private var lastUiPositionBucket = Int.MIN_VALUE
    private var lastUiFrontierBucket = Int.MIN_VALUE

    private val tick = object : Runnable {
        override fun run() {
            if (closed) return
            tickOnMain()
            mainHandler.postDelayed(this, TICK_MILLIS)
        }
    }

    init {
        synth.setVolume(DEFAULT_MIDI_VOLUME)
        prepareAudioSource()
        mainHandler.post(tick)
    }

    private fun prepareAudioSource() {
        if (originalAudioUri.scheme != ContentResolver.SCHEME_CONTENT) {
            mainHandler.post { installPlayer(originalAudioUri) }
            return
        }

        sourceFuture = sourceExecutor.submit {
            val localFile = runCatching { copySourceToCache() }.getOrNull()
            mainHandler.post {
                if (closed) {
                    localFile?.delete()
                    return@post
                }
                if (localFile != null) {
                    stagedSourceFile = localFile
                    installPlayer(Uri.fromFile(localFile))
                } else {
                    // A provider may reject stream copying while still being directly playable.
                    installPlayer(originalAudioUri)
                }
            }
        }
    }

    private fun copySourceToCache(): File {
        val directory = File(appContext.cacheDir, "muscriptor-preview").apply {
            mkdirs()
            listFiles()?.forEach { it.delete() }
        }
        val extension = sourceExtension()
        val temporary = File(directory, "source-${SystemClock.uptimeMillis()}.$extension.tmp")
        val target = File(directory, "source-${SystemClock.uptimeMillis()}.$extension")

        try {
            val input = appContext.contentResolver.openInputStream(originalAudioUri)
                ?: error("Could not open the selected audio")
            input.use { source ->
                FileOutputStream(temporary).buffered(1 shl 20).use { output ->
                    source.copyTo(output, 1 shl 20)
                }
            }
            check(temporary.renameTo(target)) { "Could not finalize preview audio" }
            return target
        } catch (error: Throwable) {
            temporary.delete()
            target.delete()
            throw error
        }
    }

    private fun sourceExtension(): String {
        val pathExtension = originalAudioUri.lastPathSegment
            ?.substringAfterLast('.', "")
            ?.lowercase()
            ?.takeIf { it.length in 1..8 && it.all(Char::isLetterOrDigit) }
        if (pathExtension != null) return pathExtension
        return MimeTypeMap.getSingleton()
            .getExtensionFromMimeType(sourceMimeType)
            ?.takeIf { it.isNotBlank() }
            ?: "audio"
    }

    private fun installPlayer(uri: Uri) {
        if (closed) return
        runCatching { player?.release() }
        started = false
        emergencyHold = false
        currentSpeed = 1f

        val mediaItem = MediaItem.Builder()
            .setUri(uri)
            .apply {
                sourceMimeType?.takeIf { it.startsWith("audio/") }?.let(::setMimeType)
            }
            .build()

        player = ExoPlayer.Builder(appContext).build().also { newPlayer ->
            newPlayer.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(C.USAGE_MEDIA)
                    .setContentType(C.AUDIO_CONTENT_TYPE_MUSIC)
                    .build(),
                false,
            )
            newPlayer.volume = requestedSongVolume
            newPlayer.addListener(object : Player.Listener {
                override fun onPlaybackStateChanged(playbackState: Int) {
                    if (playbackState == Player.STATE_ENDED) synth.allNotesOff()
                    publishState(force = true)
                }

                override fun onIsPlayingChanged(isPlaying: Boolean) {
                    publishState(force = true)
                }

                override fun onPlayerError(error: PlaybackException) {
                    synth.allNotesOff()
                    publishState(force = true)
                }
            })
            newPlayer.setMediaItem(mediaItem)
            newPlayer.prepare()
        }
        publishState(force = true)
    }

    fun accept(event: LiveNoteEvent) {
        // Starts are provisional. They stay visible in PianoRollView but are not audible until the
        // corresponding final note is emitted.
        if (event !is LiveNoteEvent.Ended) return

        val note = event.note
        if (!note.onsetSeconds.isFinite() || !note.offsetSeconds.isFinite()) return
        if (note.pitch !in 0..127) return
        if (note.offsetSeconds <= note.onsetSeconds) return

        synchronized(eventLock) {
            if (!finalizedIds.add(event.id)) return
            val on = ScheduledEvent(
                timeSeconds = note.onsetSeconds,
                priority = 1,
                sequence = eventSequence++,
                action = Action.On(
                    id = event.id,
                    program = note.program,
                    pitch = note.pitch,
                    isDrum = note.isDrum,
                    offsetSeconds = note.offsetSeconds,
                ),
            )
            val off = ScheduledEvent(
                timeSeconds = note.offsetSeconds,
                priority = 0,
                sequence = eventSequence++,
                action = Action.Off(event.id),
            )
            allEvents += on
            allEvents += off
            pendingEvents += on
            pendingEvents += off
        }
    }

    fun updateFinalizedFrontier(seconds: Double) {
        finalizedFrontierSeconds = maxOf(finalizedFrontierSeconds, seconds)
    }

    fun markInferenceComplete(durationSeconds: Double) {
        audioDurationSeconds = durationSeconds
        finalizedFrontierSeconds = maxOf(finalizedFrontierSeconds, durationSeconds)
        inferenceComplete = true
    }

    fun setSongVolume(value: Float) {
        requestedSongVolume = value.coerceIn(0f, 1f)
        runOnMain { player?.volume = requestedSongVolume }
    }

    fun setMidiVolume(value: Float) {
        synth.setVolume(value)
    }

    fun togglePause() {
        runOnMain {
            val activePlayer = player ?: return@runOnMain
            if (!started) return@runOnMain
            userPaused = !userPaused
            if (userPaused) {
                activePlayer.pause()
                synth.allNotesOff()
            } else {
                emergencyHold = false
                rebuildAt(activePlayer.currentPosition / 1000.0)
                activePlayer.play()
            }
            publishState(force = true)
        }
    }

    fun restart() {
        runOnMain {
            val activePlayer = player ?: return@runOnMain
            if (!started) return@runOnMain
            userPaused = false
            emergencyHold = false
            currentSpeed = 1f
            activePlayer.playbackParameters = PlaybackParameters(1f, 1f)
            activePlayer.seekTo(0)
            rebuildAt(0.0)
            activePlayer.play()
            publishState(force = true)
        }
    }

    private fun tickOnMain() {
        val activePlayer = player
        if (activePlayer == null) {
            val now = SystemClock.uptimeMillis()
            if (now - lastUiUpdateMs >= UI_UPDATE_MILLIS) {
                publishState(force = false)
                lastUiUpdateMs = now
            }
            return
        }

        val now = SystemClock.uptimeMillis()
        val prepared = activePlayer.playbackState == Player.STATE_READY ||
            activePlayer.playbackState == Player.STATE_ENDED

        if (!started && prepared && (inferenceComplete || finalizedFrontierSeconds >= START_BUFFER_SECONDS)) {
            started = true
            currentSpeed = 1f
            activePlayer.seekTo(0)
            rebuildAt(0.0)
            activePlayer.play()
            publishState(force = true)
        }

        val position = activePlayer.currentPosition.coerceAtLeast(0L) / 1000.0
        lastKnownPositionSeconds = position
        val frontier = if (inferenceComplete) {
            maxOf(finalizedFrontierSeconds, audioDurationSeconds)
        } else {
            finalizedFrontierSeconds
        }
        val lead = frontier - position

        if (started && !userPaused && activePlayer.playbackState != Player.STATE_ENDED) {
            if (!inferenceComplete && lead <= EMERGENCY_HOLD_SECONDS) {
                if (!emergencyHold) {
                    emergencyHold = true
                    activePlayer.pause()
                    synth.allNotesOff()
                    publishState(force = true)
                }
            } else {
                if (emergencyHold && (inferenceComplete || lead >= EMERGENCY_RESUME_SECONDS)) {
                    emergencyHold = false
                    rebuildAt(position)
                    activePlayer.play()
                    publishState(force = true)
                }

                if (!emergencyHold) {
                    val targetSpeed = if (inferenceComplete) 1f else speedForLead(lead)
                    currentSpeed += (targetSpeed - currentSpeed) * SPEED_SMOOTHING

                    if (
                        now - lastSpeedApplyMs >= SPEED_APPLY_MILLIS &&
                        abs(activePlayer.playbackParameters.speed - currentSpeed) > SPEED_CHANGE_THRESHOLD
                    ) {
                        activePlayer.playbackParameters = PlaybackParameters(currentSpeed, 1f)
                        lastSpeedApplyMs = now
                    }

                    if (!activePlayer.isPlaying && prepared) activePlayer.play()
                    processPending(position)
                }
            }
        }

        if (now - lastUiUpdateMs >= UI_UPDATE_MILLIS) {
            publishState(force = false)
            lastUiUpdateMs = now
        }
    }

    private fun speedForLead(leadSeconds: Double): Float = when {
        leadSeconds >= 4.25 -> 1.0f
        leadSeconds >= 3.0 -> (0.84 + (leadSeconds - 3.0) * 0.128).toFloat()
        leadSeconds >= 2.0 -> (0.58 + (leadSeconds - 2.0) * 0.26).toFloat()
        leadSeconds >= 1.0 -> (0.34 + (leadSeconds - 1.0) * 0.24).toFloat()
        leadSeconds >= 0.25 -> (0.12 + (leadSeconds - 0.25) * 0.293).toFloat()
        else -> MIN_PLAYBACK_SPEED
    }.coerceIn(MIN_PLAYBACK_SPEED, 1f)

    /** Drain all currently due events under one lock, then trigger the synth without holding it. */
    private fun processPending(positionSeconds: Double) {
        val audibleHorizon = positionSeconds + synth.outputLatencySeconds + EVENT_LOOKAHEAD_SECONDS
        val due = ArrayList<ScheduledEvent>(16)
        synchronized(eventLock) {
            while (true) {
                val next = pendingEvents.peek() ?: break
                if (next.timeSeconds > audibleHorizon) break
                due += pendingEvents.poll()
            }
        }

        for (event in due) {
            when (val action = event.action) {
                is Action.On -> {
                    if (action.offsetSeconds > positionSeconds - PAST_EVENT_TOLERANCE_SECONDS) {
                        synth.noteOn(
                            action.id,
                            action.program,
                            action.pitch,
                            action.isDrum,
                        )
                    }
                }
                is Action.Off -> synth.noteOff(action.id)
            }
        }
    }

    private fun rebuildAt(positionSeconds: Double) {
        synth.allNotesOff()
        lastKnownPositionSeconds = positionSeconds

        val snapshot = synchronized(eventLock) {
            pendingEvents.clear()
            allEvents.sortedWith(eventComparator)
        }
        val active = LinkedHashMap<Long, Action.On>()
        val future = ArrayList<ScheduledEvent>()

        for (event in snapshot) {
            if (event.timeSeconds <= positionSeconds) {
                when (val action = event.action) {
                    is Action.On -> active[action.id] = action
                    is Action.Off -> active.remove(action.id)
                }
            } else {
                future += event
            }
        }

        synchronized(eventLock) {
            pendingEvents.addAll(future)
        }
        active.values.forEach { action ->
            if (action.offsetSeconds > positionSeconds) {
                synth.noteOn(
                    action.id,
                    action.program,
                    action.pitch,
                    action.isDrum,
                )
            }
        }
    }

    private fun publishState(force: Boolean) {
        val activePlayer = player
        val prepared = activePlayer?.let {
            it.playbackState == Player.STATE_READY || it.playbackState == Player.STATE_ENDED
        } == true
        val position = activePlayer?.currentPosition?.coerceAtLeast(0L)?.div(1000.0)
            ?: lastKnownPositionSeconds
        val frontier = if (inferenceComplete) {
            maxOf(finalizedFrontierSeconds, audioDurationSeconds)
        } else {
            finalizedFrontierSeconds
        }
        val isPlaying = activePlayer?.isPlaying == true
        val signature =
            (if (prepared) 1 else 0) or
                (if (started) 2 else 0) or
                (if (isPlaying) 4 else 0) or
                (if (userPaused) 8 else 0) or
                (if (emergencyHold) 16 else 0)
        val positionBucket = (position * UI_POSITION_BUCKETS_PER_SECOND).toInt()
        val frontierBucket = (frontier * UI_POSITION_BUCKETS_PER_SECOND).toInt()

        if (
            !force &&
            signature == lastUiSignature &&
            positionBucket == lastUiPositionBucket &&
            frontierBucket == lastUiFrontierBucket
        ) {
            return
        }
        lastUiSignature = signature
        lastUiPositionBucket = positionBucket
        lastUiFrontierBucket = frontierBucket

        onState(
            State(
                prepared = prepared,
                started = started,
                playing = isPlaying && !userPaused && !emergencyHold,
                waitingForBuffer = !started || emergencyHold,
                positionSeconds = position,
                finalizedFrontierSeconds = frontier,
                leadSeconds = frontier - position,
                speed = if (started) currentSpeed else 0f,
            ),
        )
    }

    private fun runOnMain(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else mainHandler.post(block)
    }

    override fun close() {
        if (closed) return
        closed = true
        mainHandler.removeCallbacksAndMessages(null)
        sourceFuture?.cancel(true)
        sourceExecutor.shutdownNow()
        runCatching { player?.pause() }
        runCatching { player?.release() }
        player = null
        synth.close()
        stagedSourceFile?.delete()
        stagedSourceFile = null
    }

    companion object {
        const val TARGET_LEAD_SECONDS = 3.0
        const val DEFAULT_SONG_VOLUME = 0.20f
        const val DEFAULT_MIDI_VOLUME = 0.80f

        private const val START_BUFFER_SECONDS = 3.5
        private const val EMERGENCY_HOLD_SECONDS = 0.06
        private const val EMERGENCY_RESUME_SECONDS = 0.65
        private const val MIN_PLAYBACK_SPEED = 0.10f
        private const val SPEED_SMOOTHING = 0.12f
        private const val SPEED_CHANGE_THRESHOLD = 0.025f
        private const val SPEED_APPLY_MILLIS = 120L
        private const val EVENT_LOOKAHEAD_SECONDS = 0.008
        private const val PAST_EVENT_TOLERANCE_SECONDS = 0.015
        private const val TICK_MILLIS = 12L
        private const val UI_UPDATE_MILLIS = 100L
        private const val UI_POSITION_BUCKETS_PER_SECOND = 10.0
    }
}
