package dev.cardrhyme.muscriptormobile

import android.os.Build
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

class MuScriptorEngine(
    private val modelDir: File,
    private val maxCacheLength: Int = 1024,
    private val requestedBackend: ComputeBackend = ComputeBackend.CPU,
    private val threadCount: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 8),
) : Closeable {
    data class Progress(
        val completedChunks: Int,
        val totalChunks: Int,
        val generatedTokens: Int,
        val lastTokenMillis: Double,
        val finalizedSeconds: Double,
        val message: String,
    )

    data class Result(
        val notes: List<MidiNote>,
        val midi: ByteArray,
        val generatedTokens: Int,
    )

    private data class SessionBundle(
        val conditionerOptions: OrtSession.SessionOptions,
        val decoderOptions: OrtSession.SessionOptions,
        val conditioner: OrtSession,
        val decoder: OrtSession,
        val activeBackend: ComputeBackend,
        val status: String,
    )

    private class PreparedCondition(
        val owner: OrtSession.Result,
        val tensor: OnnxTensor,
        val length: Int,
    ) : Closeable {
        override fun close() {
            owner.close()
        }
    }

    private val environment = OrtEnvironment.getEnvironment()
    private val sessions = createSessionsWithFallback(requestedBackend)
    private val conditionerOptions = sessions.conditionerOptions
    private val decoderOptions = sessions.decoderOptions
    private val conditioner = sessions.conditioner
    private val decoder = sessions.decoder

    val activeBackend: ComputeBackend = sessions.activeBackend
    val backendStatus: String = sessions.status

    private val cache = SharedCache(environment, maxCacheLength)
    private val emptyCondition = floatTensor(FloatArray(0), longArrayOf(1, 0, MODEL_DIM.toLong()))

    /**
     * The ONNX conditioner still receives its native five-second input, but windows now advance by
     * four seconds. Each neighboring pair therefore shares one second of real audio context. Only
     * the stable center ownership region is committed, removing the artificial hard cut at 5, 10,
     * 15… seconds without increasing the conditioner tensor shape.
     */
    suspend fun transcribe(
        samples16k: FloatArray,
        onLiveEvent: (LiveNoteEvent) -> Unit,
        onProgress: (Progress) -> Unit,
    ): Result = coroutineScope {
        require(samples16k.isNotEmpty()) { "Audio contains no samples" }
        val duration = samples16k.size.toDouble() / MelSpectrogram.SAMPLE_RATE
        val totalWindows = windowCount(samples16k.size)
        val committedNotes = ArrayList<MidiNote>()
        var generatedTotal = 0
        var nextLiveId = 0L
        var finalizedFrontier = 0.0

        var prepared: PreparedCondition? = prepareCondition(samples16k, 0)
        try {
            for (windowIndex in 0 until totalWindows) {
                currentCoroutineContext().ensureActive()

                val windowStart = windowIndex * WINDOW_HOP_SECONDS
                val windowEnd = min(duration, windowStart + WINDOW_SECONDS)
                val safeStart = if (windowIndex == 0) {
                    0.0
                } else {
                    windowStart + HALF_OVERLAP_SECONDS
                }
                val safeEnd = if (windowIndex == totalWindows - 1) {
                    duration
                } else {
                    min(duration, windowStart + WINDOW_HOP_SECONDS + HALF_OVERLAP_SECONDS)
                }

                // Every overlapped window is decoded independently. Reusing the old decoder state
                // would rewind its musical clock by one second and corrupt ties/open notes.
                val windowDecoder = TokenDecoder()
                windowDecoder.startChunk(windowStart, windowEnd)
                val prompt = IntArray(0)

                onProgress(
                    Progress(
                        completedChunks = windowIndex,
                        totalChunks = totalWindows,
                        generatedTokens = generatedTotal,
                        lastTokenMillis = 0.0,
                        finalizedSeconds = finalizedFrontier,
                        message = if (activeBackend.pipelineConditioner && windowIndex + 1 < totalWindows) {
                            "Window ${windowIndex + 1}/$totalWindows • 1 s overlap • preparing next window in parallel"
                        } else {
                            "Preparing window ${windowIndex + 1}/$totalWindows • 1 s overlap"
                        },
                    ),
                )

                val nextPrepared = if (activeBackend.pipelineConditioner && windowIndex + 1 < totalWindows) {
                    async(Dispatchers.Default) {
                        prepareCondition(samples16k, windowIndex + 1)
                    }
                } else {
                    null
                }

                val currentPrepared = checkNotNull(prepared)
                prepared = null
                val windowGenerated = try {
                    runDecoder(
                        prepared = currentPrepared,
                        prompt = prompt,
                        onToken = { token, tokenMillis ->
                            windowDecoder.feed(token)
                            generatedTotal += 1
                            if (
                                generatedTotal % PROGRESS_TOKEN_INTERVAL == 0 ||
                                token == EOS_TOKEN_ID
                            ) {
                                onProgress(
                                    Progress(
                                        completedChunks = windowIndex,
                                        totalChunks = totalWindows,
                                        generatedTokens = generatedTotal,
                                        lastTokenMillis = tokenMillis,
                                        finalizedSeconds = finalizedFrontier,
                                        message = "Streaming window ${windowIndex + 1}/$totalWindows • overlapped",
                                    ),
                                )
                            }
                        },
                    )
                } finally {
                    currentPrepared.close()
                }

                generatedTotal += windowGenerated.tokensNotStreamed
                val candidates = windowDecoder.finish(windowEnd)
                val newlyCommitted = commitStableRegion(
                    candidates = candidates,
                    existing = committedNotes,
                    safeStart = safeStart,
                    safeEnd = safeEnd,
                    audioDuration = duration,
                    isLastWindow = windowIndex == totalWindows - 1,
                )
                for (note in newlyCommitted) {
                    committedNotes += note
                    val id = nextLiveId++
                    onLiveEvent(
                        LiveNoteEvent.Started(
                            id = id,
                            program = note.program,
                            pitch = note.pitch,
                            onsetSeconds = note.onsetSeconds,
                            isDrum = note.isDrum,
                        ),
                    )
                    onLiveEvent(LiveNoteEvent.Ended(id, note))
                }

                finalizedFrontier = max(finalizedFrontier, safeEnd)
                onProgress(
                    Progress(
                        completedChunks = windowIndex + 1,
                        totalChunks = totalWindows,
                        generatedTokens = generatedTotal,
                        lastTokenMillis = windowGenerated.lastTokenMillis,
                        finalizedSeconds = finalizedFrontier,
                        message = if (windowGenerated.hitCacheLimit) {
                            "Window ${windowIndex + 1}/$totalWindows complete • cache limit reached"
                        } else {
                            "Window ${windowIndex + 1}/$totalWindows complete • center committed"
                        },
                    ),
                )

                prepared = when {
                    windowIndex + 1 >= totalWindows -> null
                    nextPrepared != null -> nextPrepared.await()
                    else -> prepareCondition(samples16k, windowIndex + 1)
                }
            }
        } finally {
            prepared?.close()
        }

        val notes = mergeAdjacentForExport(committedNotes)
        Result(notes, MidiWriter.write(notes), generatedTotal)
    }

    private fun windowCount(sampleCount: Int): Int {
        if (sampleCount <= MelSpectrogram.SEGMENT_SAMPLES) return 1
        val remainingAfterFirst = sampleCount - MelSpectrogram.SEGMENT_SAMPLES
        return 1 + ceil(remainingAfterFirst.toDouble() / WINDOW_HOP_SAMPLES).toInt()
    }

    /**
     * A note is primarily owned by the window whose stable center contains its onset. If the prior
     * window missed a sustained note entirely, a center-crossing continuation is admitted at the
     * safe boundary. Nearby duplicates from onset jitter are discarded.
     */
    private fun commitStableRegion(
        candidates: List<MidiNote>,
        existing: List<MidiNote>,
        safeStart: Double,
        safeEnd: Double,
        audioDuration: Double,
        isLastWindow: Boolean,
    ): List<MidiNote> {
        val committed = ArrayList<MidiNote>()
        val ordered = candidates.sortedBy { it.onsetSeconds }

        for (raw in ordered) {
            if (!raw.onsetSeconds.isFinite() || !raw.offsetSeconds.isFinite()) continue
            if (raw.pitch !in 0..127 || raw.program !in 0..128) continue

            val onset = raw.onsetSeconds.coerceIn(0.0, audioDuration)
            val offset = raw.offsetSeconds.coerceIn(onset, audioDuration)
            if (offset - onset < MIN_ACCEPTED_NOTE_SECONDS) continue

            val ownsOnset = onset + TIME_EPSILON >= safeStart &&
                if (isLastWindow) onset <= safeEnd + TIME_EPSILON else onset < safeEnd - TIME_EPSILON

            var candidate: MidiNote? = if (ownsOnset) {
                raw.copy(onsetSeconds = onset, offsetSeconds = offset)
            } else {
                null
            }

            if (
                candidate == null &&
                !raw.isDrum &&
                onset < safeStart &&
                offset > safeStart + MIN_CONTINUATION_SECONDS
            ) {
                val covering = findCovering(existing, committed, raw, safeStart)
                candidate = when {
                    covering == null -> raw.copy(
                        onsetSeconds = safeStart,
                        offsetSeconds = offset,
                    )
                    offset > covering.offsetSeconds + MIN_CONTINUATION_SECONDS -> raw.copy(
                        onsetSeconds = max(safeStart, covering.offsetSeconds),
                        offsetSeconds = offset,
                    )
                    else -> null
                }
            }

            val note = candidate ?: continue
            if (note.offsetSeconds - note.onsetSeconds < MIN_ACCEPTED_NOTE_SECONDS) continue
            if (isNearDuplicate(existing, committed, note)) continue
            committed += note
        }
        return committed
    }

    private fun findCovering(
        existing: List<MidiNote>,
        current: List<MidiNote>,
        candidate: MidiNote,
        time: Double,
    ): MidiNote? {
        fun matches(note: MidiNote): Boolean =
            !note.isDrum &&
                note.program == candidate.program &&
                note.pitch == candidate.pitch &&
                note.onsetSeconds <= time + TIME_EPSILON &&
                note.offsetSeconds >= time - TIME_EPSILON

        for (index in current.indices.reversed()) {
            val note = current[index]
            if (matches(note)) return note
        }
        for (index in existing.indices.reversed()) {
            val note = existing[index]
            if (matches(note)) return note
            if (note.onsetSeconds < time - WINDOW_SECONDS - 0.5) break
        }
        return null
    }

    private fun isNearDuplicate(
        existing: List<MidiNote>,
        current: List<MidiNote>,
        candidate: MidiNote,
    ): Boolean {
        fun duplicate(note: MidiNote): Boolean =
            note.program == candidate.program &&
                note.pitch == candidate.pitch &&
                note.isDrum == candidate.isDrum &&
                abs(note.onsetSeconds - candidate.onsetSeconds) <= DEDUPE_ONSET_SECONDS

        for (index in current.indices.reversed()) {
            val note = current[index]
            if (note.onsetSeconds < candidate.onsetSeconds - DEDUPE_LOOKBACK_SECONDS) break
            if (duplicate(note)) return true
        }
        for (index in existing.indices.reversed()) {
            val note = existing[index]
            if (note.onsetSeconds < candidate.onsetSeconds - DEDUPE_LOOKBACK_SECONDS) break
            if (duplicate(note)) return true
        }
        return false
    }

    private fun mergeAdjacentForExport(source: List<MidiNote>): List<MidiNote> {
        val sorted = source.sortedWith(
            compareBy<MidiNote> { it.onsetSeconds }
                .thenBy { it.program }
                .thenBy { it.pitch },
        )
        val result = ArrayList<MidiNote>(sorted.size)
        val lastIndexByKey = HashMap<Triple<Int, Int, Boolean>, Int>()

        for (note in sorted) {
            val key = Triple(note.program, note.pitch, note.isDrum)
            val previousIndex = lastIndexByKey[key]
            val previous = previousIndex?.let(result::get)
            if (
                previousIndex != null &&
                previous != null &&
                !note.isDrum &&
                note.onsetSeconds <= previous.offsetSeconds + EXPORT_MERGE_GAP_SECONDS
            ) {
                result[previousIndex] = previous.copy(
                    offsetSeconds = max(previous.offsetSeconds, note.offsetSeconds),
                )
            } else {
                lastIndexByKey[key] = result.size
                result += note
            }
        }
        return result
    }

    private fun prepareCondition(
        samples16k: FloatArray,
        windowIndex: Int,
    ): PreparedCondition {
        val mel = MelSpectrogram.compute(
            samples16k,
            windowIndex * WINDOW_HOP_SAMPLES,
        )
        val melTensor = floatTensor(
            mel,
            longArrayOf(1, MelSpectrogram.FRAMES.toLong(), MelSpectrogram.N_MELS.toLong()),
        )
        val instrumentTensor = longTensor(longArrayOf(-1), longArrayOf(1, 1))
        val datasetTensor = longTensor(longArrayOf(-1), longArrayOf(1, 1))
        val conditionInputs = linkedMapOf(
            "log_mel" to melTensor,
            "instrument_ids" to instrumentTensor,
            "dataset_ids" to datasetTensor,
        )

        var result: OrtSession.Result? = null
        try {
            result = conditioner.run(conditionInputs, linkedSetOf("condition_embeddings"))
            val condition = result.get("condition_embeddings").orElseThrow() as OnnxTensor
            val shape = (condition.info as TensorInfo).shape
            return PreparedCondition(
                owner = result,
                tensor = condition,
                length = shape[1].toInt(),
            ).also { result = null }
        } finally {
            result?.close()
            melTensor.close()
            instrumentTensor.close()
            datasetTensor.close()
        }
    }

    private suspend fun runDecoder(
        prepared: PreparedCondition,
        prompt: IntArray,
        onToken: suspend (Int, Double) -> Unit,
    ): ChunkResult {
        val condition = prepared.tensor
        val conditionLength = prepared.length

        val firstIds = LongArray(prompt.size + 1)
        firstIds[0] = INITIAL_TOKEN_ID.toLong()
        prompt.forEachIndexed { index, token -> firstIds[index + 1] = token.toLong() }
        var inputIds = firstIds
        var pastLength = 0
        var generated = 0
        var lastMillis = 0.0

        val firstQueryLength = conditionLength + firstIds.size
        require(firstQueryLength <= maxCacheLength) {
            "Cache profile too small: first pass needs $firstQueryLength positions, has $maxCacheLength"
        }
        val cacheGenerationBudget = 1 + (maxCacheLength - firstQueryLength)
        val modelGenerationBudget = (MAX_SEQUENCE_TOKENS - prompt.size).coerceAtLeast(1)
        val generationBudget = min(cacheGenerationBudget, modelGenerationBudget)
        var emittedEos = false

        while (generated < generationBudget) {
            currentCoroutineContext().ensureActive()
            val prefix = if (generated == 0) condition else emptyCondition
            val prefixLength = if (generated == 0) conditionLength else 0
            val queryLength = prefixLength + inputIds.size
            val totalLength = pastLength + queryLength
            if (totalLength > maxCacheLength) break

            val inputIdTensor = longTensor(inputIds, longArrayOf(1, inputIds.size.toLong()))
            val positions = LongArray(queryLength) { index -> (pastLength + index).toLong() }
            val positionTensor = longTensor(positions, longArrayOf(1, queryLength.toLong()))
            val seqlensTensor = intTensor(intArrayOf(totalLength - 1), longArrayOf(1))
            val totalLengthTensor = intTensor(intArrayOf(totalLength), longArrayOf())
            val inputs = LinkedHashMap<String, OnnxTensor>(6 + NUM_LAYERS * 2).apply {
                put("input_ids", inputIdTensor)
                put("condition_embeddings", prefix)
                put("position_ids", positionTensor)
                put("seqlens_k", seqlensTensor)
                put("total_sequence_length", totalLengthTensor)
                putAll(cache.inputs)
            }

            val tick = System.nanoTime()
            val result = decoder.run(inputs, linkedSetOf("logits"), cache.outputs)
            lastMillis = (System.nanoTime() - tick) / 1_000_000.0
            val token = try {
                val logits = result.get("logits").orElseThrow() as OnnxTensor
                argmaxAllowed(logits)
            } finally {
                result.close()
                inputIdTensor.close()
                positionTensor.close()
                seqlensTensor.close()
                totalLengthTensor.close()
            }

            pastLength = totalLength
            if (token == EOS_TOKEN_ID) {
                emittedEos = true
                break
            }
            onToken(token, lastMillis)
            generated += 1
            inputIds = longArrayOf(token.toLong())
        }
        return ChunkResult(
            tokensNotStreamed = 0,
            lastTokenMillis = lastMillis,
            hitCacheLimit = !emittedEos && generationBudget == cacheGenerationBudget,
        )
    }

    private fun argmaxAllowed(logits: OnnxTensor): Int {
        val buffer = logits.floatBuffer ?: error("Decoder logits are not floating point")
        val limit = min(FIRST_RESERVED_TOKEN_ID, buffer.remaining())
        var bestIndex = 0
        var bestValue = -Float.MAX_VALUE
        for (index in 0 until limit) {
            val value = buffer.get(index)
            if (value.isFinite() && value > bestValue) {
                bestValue = value
                bestIndex = index
            }
        }
        check(bestValue > -Float.MAX_VALUE) { "Decoder produced no finite logits" }
        return bestIndex
    }

    private fun createSessionsWithFallback(backend: ComputeBackend): SessionBundle {
        val usesAccelerator = backend.conditionerUsesNnapi || backend.decoderUsesNnapi
        if (!usesAccelerator) return createSessions(ComputeBackend.CPU, null)
        return try {
            createSessions(backend, null)
        } catch (acceleratorError: Throwable) {
            createSessions(
                ComputeBackend.CPU,
                "Accelerator could not initialize (${compactMessage(acceleratorError)}); using CPU",
            )
        }
    }

    private fun createSessions(
        backend: ComputeBackend,
        fallbackStatus: String?,
    ): SessionBundle {
        val conditionerSessionOptions = createSessionOptions(
            useNnapi = backend.conditionerUsesNnapi,
            allowFp16 = backend.allowFp16,
            cpuThreads = if (backend.pipelineConditioner) 2 else threadCount,
        )
        val decoderSessionOptions = createSessionOptions(
            useNnapi = backend.decoderUsesNnapi,
            allowFp16 = backend.allowFp16,
            cpuThreads = threadCount,
        )

        var conditionerSession: OrtSession? = null
        var decoderSession: OrtSession? = null
        try {
            conditionerSession = environment.createSession(
                File(modelDir, "conditioner.onnx").absolutePath,
                conditionerSessionOptions,
            )
            decoderSession = environment.createSession(
                File(modelDir, "decoder.onnx").absolutePath,
                decoderSessionOptions,
            )
            val status = fallbackStatus ?: when {
                backend.pipelineConditioner -> {
                    "Parallel accelerator + CPU active • next conditioner overlaps current CPU decoder • Android chooses GPU/NPU"
                }
                backend.conditionerUsesNnapi || backend.decoderUsesNnapi -> buildString {
                    append("NNAPI accelerator active • Android chooses GPU/NPU")
                    if (backend.allowFp16) append(" • FP16 relaxation enabled")
                    append(" • unsupported nodes use ORT CPU")
                }
                else -> "CPU backend active"
            }
            return SessionBundle(
                conditionerOptions = conditionerSessionOptions,
                decoderOptions = decoderSessionOptions,
                conditioner = conditionerSession,
                decoder = decoderSession,
                activeBackend = backend,
                status = status,
            )
        } catch (error: Throwable) {
            runCatching { decoderSession?.close() }
            runCatching { conditionerSession?.close() }
            runCatching { decoderSessionOptions.close() }
            runCatching { conditionerSessionOptions.close() }
            throw error
        }
    }

    private fun createSessionOptions(
        useNnapi: Boolean,
        allowFp16: Boolean,
        cpuThreads: Int,
    ): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setInterOpNumThreads(1)
        setIntraOpNumThreads(if (useNnapi) cpuThreads.coerceAtMost(4) else cpuThreads)
        if (useNnapi) {
            require(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                "NNAPI requires Android 8.1 or newer"
            }
            val flags = EnumSet.of(NNAPIFlags.CPU_DISABLED)
            if (allowFp16) flags.add(NNAPIFlags.USE_FP16)
            addNnapi(flags)
        }
    }

    private fun compactMessage(error: Throwable): String =
        (error.message ?: error.javaClass.simpleName)
            .replace('\n', ' ')
            .take(120)

    private fun floatTensor(values: FloatArray, shape: LongArray): OnnxTensor {
        val buffer = direct(values.size * Float.SIZE_BYTES).asFloatBuffer()
        buffer.put(values)
        buffer.position(0)
        return OnnxTensor.createTensor(environment, buffer, shape)
    }

    private fun longTensor(values: LongArray, shape: LongArray): OnnxTensor {
        val buffer = direct(values.size * Long.SIZE_BYTES).asLongBuffer()
        buffer.put(values)
        buffer.position(0)
        return OnnxTensor.createTensor(environment, buffer, shape)
    }

    private fun intTensor(values: IntArray, shape: LongArray): OnnxTensor {
        val buffer = direct(values.size * Int.SIZE_BYTES).asIntBuffer()
        buffer.put(values)
        buffer.position(0)
        return OnnxTensor.createTensor(environment, buffer, shape)
    }

    private fun direct(bytes: Int): ByteBuffer =
        ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

    override fun close() {
        emptyCondition.close()
        cache.close()
        decoder.close()
        conditioner.close()
        decoderOptions.close()
        conditionerOptions.close()
    }

    private data class ChunkResult(
        val tokensNotStreamed: Int,
        val lastTokenMillis: Double,
        val hitCacheLimit: Boolean,
    )

    private class SharedCache(
        environment: OrtEnvironment,
        maxCacheLength: Int,
    ) : Closeable {
        private val tensors = ArrayList<OnnxTensor>(NUM_LAYERS * 2)
        val inputs = LinkedHashMap<String, OnnxTensor>(NUM_LAYERS * 2)
        val outputs = LinkedHashMap<String, OnnxTensor>(NUM_LAYERS * 2)

        init {
            val elements = NUM_HEADS * maxCacheLength * HEAD_DIM
            repeat(NUM_LAYERS) { layer ->
                for (kind in listOf("key", "value")) {
                    val buffer: FloatBuffer = ByteBuffer
                        .allocateDirect(elements * Float.SIZE_BYTES)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                    val tensor = OnnxTensor.createTensor(
                        environment,
                        buffer,
                        longArrayOf(1, NUM_HEADS.toLong(), maxCacheLength.toLong(), HEAD_DIM.toLong()),
                    )
                    tensors += tensor
                    inputs["past_$kind.$layer"] = tensor
                    outputs["present_$kind.$layer"] = tensor
                }
            }
        }

        override fun close() {
            tensors.forEach { it.close() }
        }
    }

    companion object {
        private const val NUM_LAYERS = 24
        private const val NUM_HEADS = 16
        private const val HEAD_DIM = 64
        private const val MODEL_DIM = 1024
        private const val INITIAL_TOKEN_ID = 1395
        private const val EOS_TOKEN_ID = 1
        private const val FIRST_RESERVED_TOKEN_ID = 1393
        private const val MAX_SEQUENCE_TOKENS = 2000

        private const val WINDOW_SECONDS = 5.0
        private const val WINDOW_HOP_SECONDS = 4.0
        private const val OVERLAP_SECONDS = WINDOW_SECONDS - WINDOW_HOP_SECONDS
        private const val HALF_OVERLAP_SECONDS = OVERLAP_SECONDS / 2.0
        private const val WINDOW_HOP_SAMPLES = 64_000

        private const val PROGRESS_TOKEN_INTERVAL = 8
        private const val TIME_EPSILON = 0.002
        private const val MIN_ACCEPTED_NOTE_SECONDS = 0.01
        private const val MIN_CONTINUATION_SECONDS = 0.04
        private const val DEDUPE_ONSET_SECONDS = 0.055
        private const val DEDUPE_LOOKBACK_SECONDS = 0.12
        private const val EXPORT_MERGE_GAP_SECONDS = 0.055
    }
}
