package dev.cardrhyme.muscriptormobile

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

/**
 * Hybrid decoder used by the native K8/V8 mode.
 *
 * The 501-condition-position prefill stays on ORT's highly optimized fused FP16 GQA path. Its cache
 * is quantized once into K8/V8. Every following one-token step uses the custom fused operator, which
 * reads the INT8 cache directly and only quantizes the newly appended K/V position.
 */
internal class K8V8Decoder(
    private val environment: OrtEnvironment,
    modelDir: File,
    private val maxCacheLength: Int,
    customOpLibraryPath: String,
    threadCount: Int,
) : Closeable {
    data class Result(
        val lastTokenMillis: Double,
        val hitCacheLimit: Boolean,
    )

    private val options = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setInterOpNumThreads(1)
        setIntraOpNumThreads(threadCount.coerceIn(2, 8))
        registerCustomOpLibrary(customOpLibraryPath)
    }
    private val session = environment.createSession(
        File(modelDir, K8V8_GRAPH).absolutePath,
        options,
    )
    private val bootstrapCacheLength = min(maxCacheLength, BOOTSTRAP_CACHE_LIMIT)
    private val bootstrapCache = Fp16Cache(environment, bootstrapCacheLength)
    private val cache = K8Cache(environment, maxCacheLength)
    private val emptyCondition = floatTensor(
        FloatArray(0),
        longArrayOf(1, 0, MODEL_DIM.toLong()),
    )

    suspend fun run(
        bootstrapSession: OrtSession,
        condition: OnnxTensor,
        conditionLength: Int,
        prompt: IntArray,
        onToken: suspend (Int, Double) -> Unit,
    ): Result {
        val firstIds = LongArray(prompt.size + 1)
        firstIds[0] = INITIAL_TOKEN_ID.toLong()
        prompt.forEachIndexed { index, token -> firstIds[index + 1] = token.toLong() }

        val firstQueryLength = conditionLength + firstIds.size
        require(firstQueryLength <= bootstrapCacheLength) {
            "K8/V8 prefill needs $firstQueryLength positions; bootstrap cache has $bootstrapCacheLength. Reduce active tie notes or use FP16 native."
        }
        require(firstQueryLength <= maxCacheLength) {
            "Cache profile too small: prefill needs $firstQueryLength positions, has $maxCacheLength"
        }

        val cacheGenerationBudget = 1 + (maxCacheLength - firstQueryLength)
        val modelGenerationBudget = (MAX_SEQUENCE_TOKENS - prompt.size).coerceAtLeast(1)
        val generationBudget = min(cacheGenerationBudget, modelGenerationBudget)

        var generated = 0
        var pastLength = 0
        var lastMillis = 0.0
        var emittedEos = false

        val prefillToken = runStep(
            decoder = bootstrapSession,
            inputIds = firstIds,
            condition = condition,
            conditionLength = conditionLength,
            pastLength = 0,
            cacheInputs = bootstrapCache.inputs,
            cacheOutputs = bootstrapCache.outputs,
        ).also { lastMillis = it.millis }.token
        pastLength = firstQueryLength

        K8V8Native.quantizeFp16Caches(
            sourceBuffers = bootstrapCache.rawBuffers,
            destinationBuffers = cache.dataBuffers,
            scaleBuffers = cache.scaleBuffers,
            sourceLength = bootstrapCacheLength,
            destinationLength = maxCacheLength,
            positions = firstQueryLength,
            heads = NUM_HEADS,
            headSize = HEAD_DIM,
        )

        if (prefillToken == EOS_TOKEN_ID) {
            emittedEos = true
        } else {
            onToken(prefillToken, lastMillis)
            generated = 1
        }

        var inputIds = longArrayOf(prefillToken.toLong())
        while (!emittedEos && generated < generationBudget) {
            currentCoroutineContext().ensureActive()
            val totalLength = pastLength + 1
            if (totalLength > maxCacheLength) break

            val step = runStep(
                decoder = session,
                inputIds = inputIds,
                condition = emptyCondition,
                conditionLength = 0,
                pastLength = pastLength,
                cacheInputs = cache.inputs,
                cacheOutputs = cache.outputs,
            )
            lastMillis = step.millis
            pastLength = totalLength
            if (step.token == EOS_TOKEN_ID) {
                emittedEos = true
                break
            }
            onToken(step.token, lastMillis)
            generated += 1
            inputIds = longArrayOf(step.token.toLong())
        }

        return Result(
            lastTokenMillis = lastMillis,
            hitCacheLimit = !emittedEos && generationBudget == cacheGenerationBudget,
        )
    }

    private data class Step(val token: Int, val millis: Double)

    private fun runStep(
        decoder: OrtSession,
        inputIds: LongArray,
        condition: OnnxTensor,
        conditionLength: Int,
        pastLength: Int,
        cacheInputs: Map<String, OnnxTensor>,
        cacheOutputs: Map<String, OnnxTensor>,
    ): Step {
        val queryLength = conditionLength + inputIds.size
        val totalLength = pastLength + queryLength
        val inputIdTensor = longTensor(inputIds, longArrayOf(1, inputIds.size.toLong()))
        val positions = LongArray(queryLength) { index -> (pastLength + index).toLong() }
        val positionTensor = longTensor(positions, longArrayOf(1, queryLength.toLong()))
        val seqlensTensor = intTensor(intArrayOf(totalLength - 1), longArrayOf(1))
        val totalLengthTensor = intTensor(intArrayOf(totalLength), longArrayOf())
        val inputs = LinkedHashMap<String, OnnxTensor>(5 + cacheInputs.size).apply {
            put("input_ids", inputIdTensor)
            put("condition_embeddings", condition)
            put("position_ids", positionTensor)
            put("seqlens_k", seqlensTensor)
            put("total_sequence_length", totalLengthTensor)
            putAll(cacheInputs)
        }

        val tick = System.nanoTime()
        val result = decoder.run(inputs, linkedSetOf("logits"), cacheOutputs)
        val millis = (System.nanoTime() - tick) / 1_000_000.0
        return try {
            val logits = result.get("logits").orElseThrow() as OnnxTensor
            Step(argmaxAllowed(logits), millis)
        } finally {
            result.close()
            inputIdTensor.close()
            positionTensor.close()
            seqlensTensor.close()
            totalLengthTensor.close()
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

    private class Fp16Cache(
        environment: OrtEnvironment,
        length: Int,
    ) : Closeable {
        private val tensors = ArrayList<OnnxTensor>(NUM_LAYERS * 2)
        private val buffers = ArrayList<ByteBuffer>(NUM_LAYERS * 2)
        val inputs = LinkedHashMap<String, OnnxTensor>(NUM_LAYERS * 2)
        val outputs = LinkedHashMap<String, OnnxTensor>(NUM_LAYERS * 2)
        val rawBuffers: Array<ByteBuffer>
            get() = buffers.toTypedArray()

        init {
            val elements = NUM_HEADS * length * HEAD_DIM
            val shape = longArrayOf(1, NUM_HEADS.toLong(), length.toLong(), HEAD_DIM.toLong())
            repeat(NUM_LAYERS) { layer ->
                for (kind in listOf("key", "value")) {
                    val bytes = direct(elements * Short.SIZE_BYTES)
                    val tensor = OnnxTensor.createTensor(
                        environment,
                        bytes.asShortBuffer(),
                        shape,
                        OnnxJavaType.FLOAT16,
                    )
                    buffers += bytes
                    tensors += tensor
                    inputs["past_$kind.$layer"] = tensor
                    outputs["present_$kind.$layer"] = tensor
                }
            }
        }

        override fun close() {
            tensors.forEach(OnnxTensor::close)
        }
    }

    private class K8Cache(
        environment: OrtEnvironment,
        length: Int,
    ) : Closeable {
        private val tensors = ArrayList<OnnxTensor>(NUM_LAYERS * 4)
        private val data = ArrayList<ByteBuffer>(NUM_LAYERS * 2)
        private val scales = ArrayList<ByteBuffer>(NUM_LAYERS * 2)
        val inputs = LinkedHashMap<String, OnnxTensor>(NUM_LAYERS * 4)
        val outputs = LinkedHashMap<String, OnnxTensor>(NUM_LAYERS * 4)
        val dataBuffers: Array<ByteBuffer>
            get() = data.toTypedArray()
        val scaleBuffers: Array<ByteBuffer>
            get() = scales.toTypedArray()

        init {
            val dataElements = NUM_HEADS * length * HEAD_DIM
            val scaleElements = NUM_HEADS * length
            val dataShape = longArrayOf(1, NUM_HEADS.toLong(), length.toLong(), HEAD_DIM.toLong())
            val scaleShape = longArrayOf(1, NUM_HEADS.toLong(), length.toLong())

            repeat(NUM_LAYERS) { layer ->
                for (kind in listOf("key", "value")) {
                    val dataBytes = direct(dataElements)
                    val scaleBytes = direct(scaleElements * Float.SIZE_BYTES)
                    val dataTensor = OnnxTensor.createTensor(
                        environment,
                        dataBytes,
                        dataShape,
                        OnnxJavaType.INT8,
                    )
                    val scaleTensor = OnnxTensor.createTensor(
                        environment,
                        scaleBytes.asFloatBuffer(),
                        scaleShape,
                    )
                    data += dataBytes
                    scales += scaleBytes
                    tensors += dataTensor
                    tensors += scaleTensor
                    inputs["past_$kind.$layer"] = dataTensor
                    inputs["past_${kind}_scale.$layer"] = scaleTensor
                    outputs["present_$kind.$layer"] = dataTensor
                    outputs["present_${kind}_scale.$layer"] = scaleTensor
                }
            }
        }

        override fun close() {
            tensors.forEach(OnnxTensor::close)
        }
    }

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

    override fun close() {
        emptyCondition.close()
        cache.close()
        bootstrapCache.close()
        session.close()
        options.close()
    }

    companion object {
        private const val K8V8_GRAPH = "decoder-cache-k8v8-native.onnx"
        private const val BOOTSTRAP_CACHE_LIMIT = 768
        private const val NUM_LAYERS = 24
        private const val NUM_HEADS = 16
        private const val HEAD_DIM = 64
        private const val MODEL_DIM = 1024
        private const val INITIAL_TOKEN_ID = 1395
        private const val EOS_TOKEN_ID = 1
        private const val FIRST_RESERVED_TOKEN_ID = 1393
        private const val MAX_SEQUENCE_TOKENS = 2000

        private fun direct(bytes: Int): ByteBuffer =
            ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
    }
}
