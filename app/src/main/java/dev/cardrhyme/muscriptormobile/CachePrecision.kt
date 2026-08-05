package dev.cardrhyme.muscriptormobile

import kotlin.math.floor

/** Persistent KV-cache storage formats. */
enum class CachePrecision(
    val displayName: String,
    val shortName: String,
    val bytesNumerator: Int,
    val bytesDenominator: Int,
    val assetFileName: String?,
    val storage: Storage,
    val qualityNote: String,
    val usesFullCacheBoundaryConversion: Boolean,
    private val extraBytesPerPosition: Long = 0,
) {
    FP32(
        displayName = "FP32 · exact / baseline",
        shortName = "FP32",
        bytesNumerator = 4,
        bytesDenominator = 1,
        assetFileName = null,
        storage = Storage.FP32,
        qualityNote = "Original fused GQA cache; fastest reliable baseline.",
        usesFullCacheBoundaryConversion = false,
    ),
    FP16(
        displayName = "FP16 native GQA · recommended",
        shortName = "FP16 native",
        bytesNumerator = 2,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-fp16-native-gqa.onnx",
        storage = Storage.FP16,
        qualityNote = "Fused attention reads and writes FP16 KV directly; only small QKV activations are cast.",
        usesFullCacheBoundaryConversion = false,
    ),
    BF16(
        displayName = "BF16 compatibility · slow",
        shortName = "BF16 compat",
        bytesNumerator = 2,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-bf16.onnx",
        storage = Storage.BF16,
        qualityNote = "Legacy high-range cache with full-cache conversion around attention.",
        usesFullCacheBoundaryConversion = true,
    ),
    INT8_BALANCED(
        displayName = "INT8 · balanced / memory only",
        shortName = "INT8",
        bytesNumerator = 1,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-int8-balanced.onnx",
        storage = Storage.INT8,
        qualityNote = "4× smaller than FP32; fixed ±4 range with 1/32 steps.",
        usesFullCacheBoundaryConversion = true,
    ),
    INT8_WIDE(
        displayName = "INT8 · wide / memory only",
        shortName = "INT8 wide",
        bytesNumerator = 1,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-int8-wide.onnx",
        storage = Storage.INT8,
        qualityNote = "4× smaller than FP32; fixed ±8 range with 1/16 steps.",
        usesFullCacheBoundaryConversion = true,
    ),
    INT4_BALANCED(
        displayName = "INT4 packed · balanced / memory only",
        shortName = "INT4",
        bytesNumerator = 1,
        bytesDenominator = 2,
        assetFileName = "decoder-cache-int4-balanced.onnx",
        storage = Storage.PACKED_UINT4,
        qualityNote = "8× smaller than FP32; experimental ±4 range with 0.5 steps.",
        usesFullCacheBoundaryConversion = true,
    ),
    INT4_WIDE(
        displayName = "INT4 packed · wide / memory only",
        shortName = "INT4 wide",
        bytesNumerator = 1,
        bytesDenominator = 2,
        assetFileName = "decoder-cache-int4-wide.onnx",
        storage = Storage.PACKED_UINT4,
        qualityNote = "8× smaller than FP32; experimental ±8 range with 1.0 steps.",
        usesFullCacheBoundaryConversion = true,
    ),
    FP16_COMPAT(
        displayName = "FP16 compatibility · legacy slow",
        shortName = "FP16 compat",
        bytesNumerator = 2,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-fp16.onnx",
        storage = Storage.FP16,
        qualityNote = "Legacy half-size cache that converts the complete cache around attention.",
        usesFullCacheBoundaryConversion = true,
    ),
    K8V8_NATIVE(
        displayName = "K8/V8 native fused · experimental",
        shortName = "K8/V8 native",
        bytesNumerator = 1,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-k8v8-native.onnx",
        storage = Storage.K8V8,
        qualityNote = "Fused INT8 attention reads K/V directly; per-head/per-position FP32 scales are included in the budget.",
        usesFullCacheBoundaryConversion = false,
        // 24 layers × K/V × 16 heads × one FP32 scale.
        extraBytesPerPosition = 3_072L,
    );

    enum class Storage {
        FP32,
        FP16,
        BF16,
        INT8,
        PACKED_UINT4,
        K8V8,
    }

    val packedHeadDim: Int
        get() = if (storage == Storage.PACKED_UINT4) HEAD_DIM / 2 else HEAD_DIM

    val bytesPerPosition: Long
        get() = ELEMENTS_PER_POSITION * bytesNumerator / bytesDenominator + extraBytesPerPosition

    fun cacheLengthForBudget(memoryMiB: Int): Int {
        val budgetBytes = memoryMiB.toLong() * MIB_BYTES
        return floor(budgetBytes.toDouble() / bytesPerPosition)
            .toInt()
            .coerceIn(1, MODEL_MAX_CACHE_LENGTH)
    }

    fun actualMemoryMiB(cacheLength: Int): Double =
        cacheLength.toDouble() * bytesPerPosition / MIB_BYTES.toDouble()

    fun estimatedGenerationPositions(cacheLength: Int): Int =
        (cacheLength - ESTIMATED_CONDITION_POSITIONS - 1).coerceAtLeast(0)

    companion object {
        const val MODEL_MAX_CACHE_LENGTH = 2504
        const val ESTIMATED_CONDITION_POSITIONS = 501
        const val HEAD_DIM = 64

        private const val NUM_LAYERS = 24L
        private const val NUM_KV_TENSORS_PER_LAYER = 2L
        private const val NUM_HEADS = 16L
        private const val HEAD_DIM_LONG = 64L
        private const val ELEMENTS_PER_POSITION =
            NUM_LAYERS * NUM_KV_TENSORS_PER_LAYER * NUM_HEADS * HEAD_DIM_LONG
        private const val MIB_BYTES = 1024L * 1024L
    }
}
