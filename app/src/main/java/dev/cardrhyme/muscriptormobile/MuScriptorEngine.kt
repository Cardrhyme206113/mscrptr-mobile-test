package dev.cardrhyme.muscriptormobile

import android.os.Build
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.ceil
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
        val message: String,
    )

    data class Result(
        val notes: List<MidiNote>,
        val midi: ByteArray,
        val generatedTokens: Int,
    )

    private data class SessionBundle(
        val options: OrtSession.SessionOptions,
        val conditioner: OrtSession,
        val decoder: OrtSession,
        val activeBackend: ComputeBackend,
        val status: String,
    )

    private val environment = OrtEnvironment.getEnvironment()
    private val sessions = createSessionsWithFallback(requestedBackend)
    private val options = sessions.options
    private val conditioner = sessions.conditioner
    private val decoder = sessions.decoder

    val activeBackend: ComputeBackend = sessions.activeBackend
    val backendStatus: String = sessions.status

    private val cache = SharedCache(environment, maxCacheLength)
    private val emptyCondition = floatTensor(FloatArray(0), longArrayOf(1, 0, MODEL_DIM.toLong()))

    suspend fun transcribe(
        samples16k: FloatArray,
        onLiveEvent: (LiveNoteEvent) -> Unit,
        onProgress: (Progress) -> Unit,
    ): Result {
        require(samples16k.isNotEmpty()) { "Audio contains no samples" }
        val duration = samples16k.size.toDouble() / MelSpectrogram.SAMPLE_RATE
        val totalChunks = ceil(samples16k.size.toDouble() / MelSpectrogram.SEGMENT_SAMPLES).toInt()
        val tokenDecoder = TokenDecoder(onLiveEvent = onLiveEvent)
        var generatedTotal = 0

        for (chunkIndex in 0 until totalChunks) {
            currentCoroutineContext().ensureActive()
            val seek = chunkIndex * 5.0
            val nextSeek = if (chunkIndex + 1 < totalChunks) (chunkIndex + 1) * 5.0 else null
            tokenDecoder.startChunk(seek, nextSeek)

            val prompt = if (chunkIndex == 0) IntArray(0) else tiePrompt(tokenDecoder.openKeys())
            prompt.forEach(tokenDecoder::feed)

            onProgress(
                Progress(
                    completedChunks = chunkIndex,
                    totalChunks = totalChunks,
                    generatedTokens = generatedTotal,
                    lastTokenMillis = 0.0,
                    message = "Preparing chunk ${chunkIndex + 1}/$totalChunks",
                ),
            )

            val mel = MelSpectrogram.compute(
                samples16k,
                chunkIndex * MelSpectrogram.SEGMENT_SAMPLES,
            )
            val chunkGenerated = runChunk(
                mel = mel,
                prompt = prompt,
                onToken = { token, tokenMillis ->
                    tokenDecoder.feed(token)
                    generatedTotal += 1
                    onProgress(
                        Progress(
                            completedChunks = chunkIndex,
                            totalChunks = totalChunks,
                            generatedTokens = generatedTotal,
                            lastTokenMillis = tokenMillis,
                            message = "Streaming chunk ${chunkIndex + 1}/$totalChunks",
                        ),
                    )
                },
            )
            generatedTotal += chunkGenerated.tokensNotStreamed
            onProgress(
                Progress(
                    completedChunks = chunkIndex + 1,
                    totalChunks = totalChunks,
                    generatedTokens = generatedTotal,
                    lastTokenMillis = chunkGenerated.lastTokenMillis,
                    message = if (chunkGenerated.hitCacheLimit) {
                        "Chunk ${chunkIndex + 1}/$totalChunks complete • cache limit reached"
                    } else {
                        "Chunk ${chunkIndex + 1}/$totalChunks complete"
                    },
                ),
            )
        }

        val notes = tokenDecoder.finish(duration)
        return Result(notes, MidiWriter.write(notes), generatedTotal)
    }

    private suspend fun runChunk(
        mel: FloatArray,
        prompt: IntArray,
        onToken: suspend (Int, Double) -> Unit,
    ): ChunkResult {
        val melTensor = floatTensor(mel, longArrayOf(1, MelSpectrogram.FRAMES.toLong(), MelSpectrogram.N_MELS.toLong()))
        val instrumentTensor = longTensor(longArrayOf(-1), longArrayOf(1, 1))
        val datasetTensor = longTensor(longArrayOf(-1), longArrayOf(1, 1))
        val conditionInputs = linkedMapOf(
            "log_mel" to melTensor,
            "instrument_ids" to instrumentTensor,
            "dataset_ids" to datasetTensor,
        )

        var conditionResult: OrtSession.Result? = null
        try {
            conditionResult = conditioner.run(conditionInputs, linkedSetOf("condition_embeddings"))
            val condition = conditionResult.get("condition_embeddings").orElseThrow() as OnnxTensor
            val shape = (condition.info as TensorInfo).shape
            val conditionLength = shape[1].toInt()

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
        } finally {
            conditionResult?.close()
            melTensor.close()
            instrumentTensor.close()
            datasetTensor.close()
        }
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

    private fun tiePrompt(openKeys: List<Pair<Int, Int>>): IntArray {
        val tokens = ArrayList<Int>(openKeys.size * 2 + 1)
        var previousProgram: Int? = null
        for ((program, pitch) in openKeys) {
            if (program != previousProgram) {
                tokens += PROGRAM_TOKEN_START + program
                previousProgram = program
            }
            tokens += PITCH_TOKEN_START + pitch
        }
        tokens += TIE_TOKEN_ID
        return tokens.toIntArray()
    }

    private fun createSessionsWithFallback(backend: ComputeBackend): SessionBundle {
        if (!backend.useNnapi) return createSessions(ComputeBackend.CPU, null)
        return try {
            createSessions(backend, null)
        } catch (nnapiError: Throwable) {
            createSessions(
                ComputeBackend.CPU,
                "NNAPI could not initialize (${compactMessage(nnapiError)}); using CPU",
            )
        }
    }

    private fun createSessions(
        backend: ComputeBackend,
        fallbackStatus: String?,
    ): SessionBundle {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setInterOpNumThreads(1)
            setIntraOpNumThreads(
                if (backend.useNnapi) threadCount.coerceAtMost(4) else threadCount,
            )
            if (backend.useNnapi) {
                require(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                    "NNAPI requires Android 8.1 or newer"
                }
                val flags = EnumSet.of(NNAPIFlags.CPU_DISABLED)
                if (backend.allowFp16) flags.add(NNAPIFlags.USE_FP16)
                addNnapi(flags)
            }
        }

        var conditionerSession: OrtSession? = null
        try {
            conditionerSession = environment.createSession(
                File(modelDir, "conditioner.onnx").absolutePath,
                sessionOptions,
            )
            val decoderSession = environment.createSession(
                File(modelDir, "decoder.onnx").absolutePath,
                sessionOptions,
            )
            val status = fallbackStatus ?: if (backend.useNnapi) {
                buildString {
                    append("NNAPI hybrid active")
                    if (backend.allowFp16) append(" • FP16 relaxation enabled")
                    append(" • unsupported nodes use ORT CPU")
                }
            } else {
                "CPU backend active"
            }
            return SessionBundle(
                options = sessionOptions,
                conditioner = conditionerSession,
                decoder = decoderSession,
                activeBackend = backend,
                status = status,
            )
        } catch (error: Throwable) {
            runCatching { conditionerSession?.close() }
            runCatching { sessionOptions.close() }
            throw error
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
        options.close()
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
        private const val PITCH_TOKEN_START = 1004
        private const val TIE_TOKEN_ID = 1134
        private const val PROGRAM_TOKEN_START = 1135
    }
}
