package dev.cardrhyme.muscriptormobile

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
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
 * Uses the original recording as the master clock and schedules generated MIDI against it.
 * Playback normally stays three seconds behind fully generated chunks. When inference falls
 * behind, ExoPlayer is slowed with pitch fixed at 1.0; only a completely empty safety buffer
 * causes a short emergency hold.
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
        ) : Action

        data class Off(val id: Long) : Action
    }

    private data class ScheduledEvent(
        val timeSeconds: Double,
        val priority: Int,
        val sequence: Long,
        val action: Action,
    )

    private val mainHandler = Handler(Looper.getMainLooper())
    private val eventLock = Any()
    private val allEvents = ArrayList<ScheduledEvent>()
    private val pendingEvents = PriorityQueue<ScheduledEvent>(
        compareBy<ScheduledEvent> { it.timeSeconds }
            .thenBy { it.priority }
            .thenBy { it.sequence },
    )
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

    private var closed = false
    private var started = false
    private var userPaused = false
    private var emergencyHold = false
    private var currentSpeed = 1f

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
        val scheduled = when (event) {
            is LiveNoteEvent.Started -> ScheduledEvent(
                timeSeconds = event.onsetSeconds,
                priority = 1,
                sequence = nextSequence(),
                action = Action.On(
                    id = event.id,
                    program = event.program,
                    pitch = event.pitch,
                    isDrum = event.isDrum,
                ),
            )
            is LiveNoteEvent.Ended -> ScheduledEvent(
                timeSeconds = event.note.offsetSeconds,
                priority = 0,
                sequence = nextSequence(),
                action = Action.Off(event.id),
            )
        }
        synchronized(eventLock) {
            allEvents += scheduled
            pendingEvents += scheduled
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
        }
    }

    fun restart() {
        runOnMain {
            if (!started) return@runOnMain
            userPaused = false
            emergencyHold = false
            player.seekTo(0)
            rebuildAt(0.0)
            player.play()
        }
    }

    private fun tickOnMain() {
        val prepared = player.playbackState == Player.STATE_READY || player.playbackState == Player.STATE_ENDED
        if (!started && prepared && (inferenceComplete || finalizedFrontierSeconds >= START_BUFFER_SECONDS)) {
            started = true
            player.seekTo(0)
            rebuildAt(0.0)
            player.play()
        }

        val position = player.currentPosition.coerceAtLeast(0L) / 1000.0
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
                }
            } else {
                if (emergencyHold && (inferenceComplete || lead >= EMERGENCY_RESUME_SECONDS)) {
                    emergencyHold = false
                    rebuildAt(position)
                    player.play()
                }
                if (!emergencyHold) {
                    val targetSpeed = if (inferenceComplete) 1f else speedForLead(lead)
                    currentSpeed += (targetSpeed - currentSpeed) * SPEED_SMOOTHING
                    if (abs(player.playbackParameters.speed - currentSpeed) > 0.015f) {
                        // pitch=1 keeps musical pitch stable while the recording slows down.
                        player.playbackParameters = PlaybackParameters(currentSpeed, 1f)
                    }
                    if (!player.isPlaying && prepared) player.play()
                    processPending(position)
                }
            }
        }

        onState(
            State(
                prepared = prepared,
                started = started,
                playing = player.isPlaying && !userPaused && !emergencyHold,
                waitingForBuffer = !started || emergencyHold,
                positionSeconds = position,
                finalizedFrontierSeconds = frontier,
                leadSeconds = lead,
                speed = if (started) currentSpeed else 0f,
            ),
        )
    }

    private fun speedForLead(leadSeconds: Double): Float = when {
        leadSeconds >= 4.25 -> 1.0f
        leadSeconds >= 3.0 -> (0.84 + (leadSeconds - 3.0) * 0.128).toFloat()
        leadSeconds >= 2.0 -> (0.58 + (leadSeconds - 2.0) * 0.26).toFloat()
        leadSeconds >= 1.0 -> (0.34 + (leadSeconds - 1.0) * 0.24).toFloat()
        leadSeconds >= 0.25 -> (0.12 + (leadSeconds - 0.25) * 0.293).toFloat()
        else -> MIN_PLAYBACK_SPEED
    }.coerceIn(MIN_PLAYBACK_SPEED, 1f)

    private fun processPending(positionSeconds: Double) {
        while (true) {
            val event = synchronized(eventLock) {
                val next = pendingEvents.peek() ?: return
                if (next.timeSeconds <= positionSeconds + EVENT_LOOKAHEAD_SECONDS) pendingEvents.poll() else return
            }
            when (val action = event.action) {
                is Action.On -> synth.noteOn(
                    action.id,
                    action.program,
                    action.pitch,
                    action.isDrum,
                )
                is Action.Off -> synth.noteOff(action.id)
            }
        }
    }

    private fun rebuildAt(positionSeconds: Double) {
        synth.allNotesOff()
        val active = LinkedHashMap<Long, Action.On>()
        synchronized(eventLock) {
            pendingEvents.clear()
            for (event in allEvents.sortedWith(
                compareBy<ScheduledEvent> { it.timeSeconds }
                    .thenBy { it.priority }
                    .thenBy { it.sequence },
            )) {
                if (event.timeSeconds <= positionSeconds) {
                    when (val action = event.action) {
                        is Action.On -> active[action.id] = action
                        is Action.Off -> active.remove(action.id)
                    }
                } else {
                    pendingEvents += event
                }
            }
        }
        active.values.forEach {
            synth.noteOn(it.id, it.program, it.pitch, it.isDrum)
        }
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
        private const val SPEED_SMOOTHING = 0.18f
        private const val EVENT_LOOKAHEAD_SECONDS = 0.018
        private const val TICK_MILLIS = 40L
    }
}
