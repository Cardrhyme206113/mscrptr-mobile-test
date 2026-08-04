package dev.cardrhyme.muscriptormobile

import kotlin.math.floor

/**
 * Persistent KV-cache storage formats.
 *
 * FP32 uses the original decoder graph. The other formats use tiny graph adapters generated at
 * build time. They dequantize one layer's cache for attention and quantize it again at the output,
 * while the long-lived shared cache remains in the selected compact representation.
 */
enum class CachePrecision(
    val displayName: String,
    val shortName: String,
    val bytesNumerator: Int,
    val bytesDenominator: Int,
    val assetFileName: String?,
    val storage: Storage,
    val qualityNote: String,
) {
    FP32(
        displayName = "FP32 · exact",
        shortName = "FP32",
        bytesNumerator = 4,
        bytesDenominator = 1,
        assetFileName = null,
        storage = Storage.FP32,
        qualityNote = "Original cache precision; highest memory use.",
    ),
    FP16(
        displayName = "FP16 · high quality",
        shortName = "FP16",
        bytesNumerator = 2,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-fp16.onnx",
        storage = Storage.FP16,
        qualityNote = "Half-size IEEE cache; normally near-FP32 quality.",
    ),
    BF16(
        displayName = "BF16 · high range",
        shortName = "BF16",
        bytesNumerator = 2,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-bf16.onnx",
        storage = Storage.BF16,
        qualityNote = "Half-size cache with FP32-like exponent range and lower mantissa precision.",
    ),
    INT8_BALANCED(
        displayName = "INT8 · balanced",
        shortName = "INT8",
        bytesNumerator = 1,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-int8-balanced.onnx",
        storage = Storage.INT8,
        qualityNote = "4× smaller than FP32; fixed ±4 range with 1/32 steps.",
    ),
    INT8_WIDE(
        displayName = "INT8 · wide range",
        shortName = "INT8 wide",
        bytesNumerator = 1,
        bytesDenominator = 1,
        assetFileName = "decoder-cache-int8-wide.onnx",
        storage = Storage.INT8,
        qualityNote = "4× smaller than FP32; fixed ±8 range with 1/16 steps.",
    ),
    INT4_BALANCED(
        displayName = "INT4 packed · balanced",
        shortName = "INT4",
        bytesNumerator = 1,
        bytesDenominator = 2,
        assetFileName = "decoder-cache-int4-balanced.onnx",
        storage = Storage.PACKED_UINT4,
        qualityNote = "8× smaller than FP32; experimental ±4 range with 0.5 steps.",
    ),
    INT4_WIDE(
        displayName = "INT4 packed · wide range",
        shortName = "INT4 wide",
        bytesNumerator = 1,
        bytesDenominator = 2,
        assetFileName = "decoder-cache-int4-wide.onnx",
        storage = Storage.PACKED_UINT4,
        qualityNote = "8× smaller than FP32; experimental ±8 range with 1.0 steps.",
    );

    enum class Storage {
        FP32,
        FP16,
        BF16,
        INT8,
        PACKED_UINT4,
    }

    val packedHeadDim: Int
        get() = if (storage == Storage.PACKED_UINT4) HEAD_DIM / 2 else HEAD_DIM

    val bytesPerPosition: Long
        get() = ELEMENTS_PER_POSITION * bytesNumerator / bytesDenominator

    fun cacheLengthForBudget(memoryMiB: Int): Int {
        val budgetBytes = memoryMiB.toLong() * MIB
        return floor(budgetBytes.toDouble() / bytesPerPosition)
            .toInt()
            .coerceIn(1, MODEL_MAX_CACHE_LENGTH)
    }

    fun actualMemoryMiB(cacheLength: Int): Double =
        cacheLength.toDouble() * bytesPerPosition / MIB

    fun estimatedGenerationPositions(cacheLength: Int): Int =
        (cacheLength - ESTIMATED_CONDITION_POSITIONS - 1).coerceAtLeast(0)

    companion object {
        const val MODEL_MAX_CACHE_LENGTH = 2504
        const val ESTIMATED_CONDITION_POSITIONS = 501
        const val HEAD_DIM = 64

        private const val NUM_LAYERS = 24L
        private const val NUM_KV_TENSORS_PER_LAYER = 2L
        private const val NUM_HEADS = 16L
        private const val ELEMENTS_PER_POSITION =
            NUM_LAYERS * NUM_KV_TENSORS_PER_LAYER * NUM_HEADS * HEAD_DIM
        private const val MIB = 1024.0 * 1024.0
    }
}
