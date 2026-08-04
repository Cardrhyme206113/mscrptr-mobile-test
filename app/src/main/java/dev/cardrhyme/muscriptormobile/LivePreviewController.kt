package dev.cardrhyme.muscriptormobile

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.media3.common.AudioAttributes
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import java.io.Closeable
import java.util.PriorityQueue
import kotlin.math.abs

/**
 * Uses the original recording as the master clock and schedules finalized MIDI notes against it.
 *
 * LiveNoteEvent.Started is intentionally not sent to the synth: it is provisional until the model
 * emits the matching Ended event. Playing provisional starts caused stale/random notes that never
 * existed in the exported MIDI. The piano roll may still display provisional notes separately.
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

    private val mainHandler = Handler(Looper.getMainLooper())
    private val eventLock = Any()
    private val allEvents = ArrayList<ScheduledEvent>()
    private val pendingEvents = PriorityQueue(eventComparator)
    private val finalizedIds = HashSet<Long>()
    private var eventSequence = 0L

    private val synth = LiveMidiSynth()
    private val player = ExoPlayer.Builder(context).build().apply {
        setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(C.USAGE_MEDIA)
                .setContentType(C.AUDIO_CONTENT_TYPE_MUSIC)
                .build(),
            false,
        )
        setMediaItem(MediaItem.fromUri(audioUri))
        volume = DEFAULT_SONG_VOLUME
        addListener(object : Player.Listener {
            override fun onPlaybackStateChanged(playbackState: Int) {
                if (playbackState == Player.STATE_ENDED) synth.allNotesOff()
            }
        })
        prepare()
    }

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

    private val tick = object : Runnable {
        override fun run() {
            if (closed) return
            tickOnMain()
            mainHandler.postDelayed(this, TICK_MILLIS)
        }
    }

    init {
        synth.setVolume(DEFAULT_MIDI_VOLUME)
        mainHandler.post(tick)
    }

    fun accept(event: LiveNoteEvent) {
        // Starts are provisional. They stay visible in PianoRollView but are not audible until the
        // corresponding final note is emitted.
        if (event !is LiveNoteEvent.Ended) return

        val note = event.note
        if (!note.onsetSeconds.isFinite() || !note.offsetSeconds.isFinite()) return
        if (note.pitch !in 0..127) return
        if (note.offsetSeconds <= note.onsetSeconds) return

        val on = ScheduledEvent(
            timeSeconds = note.onsetSeconds,
            priority = 1,
            sequence = nextSequence(),
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
            sequence = nextSequence(),
            action = Action.Off(event.id),
        )

        synchronized(eventLock) {
            if (!finalizedIds.add(event.id)) return
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
        runOnMain { player.volume = value.coerceIn(0f, 1f) }
    }

    fun setMidiVolume(value: Float) {
        synth.setVolume(value)
    }

    fun togglePause() {
        runOnMain {
            if (!started) return@runOnMain
            userPaused = !userPaused
            if (userPaused) {
                player.pause()
                synth.allNotesOff()
            } else {
                emergencyHold = false
                rebuildAt(player.currentPosition / 1000.0)
                player.play()
            }
            publishState(force = true)
        }
    }

    fun restart() {
        runOnMain {
            if (!started) return@runOnMain
            userPaused = false
            emergencyHold = false
            currentSpeed = 1f
            player.playbackParameters = PlaybackParameters(1f, 1f)
            player.seekTo(0)
            rebuildAt(0.0)
            player.play()
            publishState(force = true)
        }
    }

    private fun tickOnMain() {
        val now = SystemClock.uptimeMillis()
        val prepared = player.playbackState == Player.STATE_READY ||
            player.playbackState == Player.STATE_ENDED

        if (!started && prepared && (inferenceComplete || finalizedFrontierSeconds >= START_BUFFER_SECONDS)) {
            started = true
            currentSpeed = 1f
            player.seekTo(0)
            rebuildAt(0.0)
            player.play()
            publishState(force = true)
        }

        val position = player.currentPosition.coerceAtLeast(0L) / 1000.0
        lastKnownPositionSeconds = position
        val frontier = if (inferenceComplete) {
            maxOf(finalizedFrontierSeconds, audioDurationSeconds)
        } else {
            finalizedFrontierSeconds
        }
        val lead = frontier - position

        if (started && !userPaused && player.playbackState != Player.STATE_ENDED) {
            if (!inferenceComplete && lead <= EMERGENCY_HOLD_SECONDS) {
                if (!emergencyHold) {
                    emergencyHold = true
                    player.pause()
                    synth.allNotesOff()
                    publishState(force = true)
                }
            } else {
                if (emergencyHold && (inferenceComplete || lead >= EMERGENCY_RESUME_SECONDS)) {
                    emergencyHold = false
                    rebuildAt(position)
                    player.play()
                    publishState(force = true)
                }

                if (!emergencyHold) {
                    val targetSpeed = if (inferenceComplete) 1f else speedForLead(lead)
                    currentSpeed += (targetSpeed - currentSpeed) * SPEED_SMOOTHING

                    if (
                        now - lastSpeedApplyMs >= SPEED_APPLY_MILLIS &&
                        abs(player.playbackParameters.speed - currentSpeed) > SPEED_CHANGE_THRESHOLD
                    ) {
                        // pitch=1 keeps musical pitch stable while the recording slows down.
                        player.playbackParameters = PlaybackParameters(currentSpeed, 1f)
                        lastSpeedApplyMs = now
                    }

                    if (!player.isPlaying && prepared) player.play()
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

    /**
     * Trigger notes early by the AudioTrack queue depth so their audible output aligns with the
     * original song rather than arriving one hardware buffer late.
     */
    private fun processPending(positionSeconds: Double) {
        val audibleHorizon = positionSeconds + synth.outputLatencySeconds + EVENT_LOOKAHEAD_SECONDS

        while (true) {
            val event = synchronized(eventLock) {
                val next = pendingEvents.peek() ?: return
                if (next.timeSeconds <= audibleHorizon) pendingEvents.poll() else return
            }

            when (val action = event.action) {
                is Action.On -> {
                    // A finalized long note may arrive after its onset. Start it late only while it
                    // is still active; never burst notes whose entire lifetime is already behind us.
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
        val prepared = player.playbackState == Player.STATE_READY ||
            player.playbackState == Player.STATE_ENDED
        val position = player.currentPosition.coerceAtLeast(0L) / 1000.0
        val frontier = if (inferenceComplete) {
            maxOf(finalizedFrontierSeconds, audioDurationSeconds)
        } else {
            finalizedFrontierSeconds
        }
        val signature =
            (if (prepared) 1 else 0) or
                (if (started) 2 else 0) or
                (if (player.isPlaying) 4 else 0) or
                (if (userPaused) 8 else 0) or
                (if (emergencyHold) 16 else 0)

        if (!force && signature == lastUiSignature && closed) return
        lastUiSignature = signature

        onState(
            State(
                prepared = prepared,
                started = started,
                playing = player.isPlaying && !userPaused && !emergencyHold,
                waitingForBuffer = !started || emergencyHold,
                positionSeconds = position,
                finalizedFrontierSeconds = frontier,
                leadSeconds = frontier - position,
                speed = if (started) currentSpeed else 0f,
            ),
        )
    }

    private fun nextSequence(): Long = synchronized(eventLock) { eventSequence++ }

    private fun runOnMain(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else mainHandler.post(block)
    }

    override fun close() {
        if (closed) return
        closed = true
        mainHandler.removeCallbacksAndMessages(null)
        runCatching { player.pause() }
        runCatching { player.release() }
        synth.close()
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
    }
}
