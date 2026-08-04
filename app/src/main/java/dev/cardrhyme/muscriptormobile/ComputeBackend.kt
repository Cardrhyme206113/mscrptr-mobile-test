package dev.cardrhyme.muscriptormobile

/** Runtime backend requested for ONNX Runtime sessions. */
enum class ComputeBackend(
    val displayName: String,
    val shortName: String,
    val useNnapi: Boolean,
    val allowFp16: Boolean,
) {
    CPU(
        displayName = "CPU • reliable",
        shortName = "CPU",
        useNnapi = false,
        allowFp16 = false,
    ),
    NNAPI(
        displayName = "NPU / NNAPI • experimental",
        shortName = "NNAPI",
        useNnapi = true,
        allowFp16 = false,
    ),
    NNAPI_FP16(
        displayName = "NPU / NNAPI + FP16 • fastest experiment",
        shortName = "NNAPI FP16",
        useNnapi = true,
        allowFp16 = true,
    ),
}
