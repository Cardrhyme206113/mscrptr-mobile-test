package dev.cardrhyme.muscriptormobile

/** Runtime layout requested for the ONNX Runtime conditioner and decoder sessions. */
enum class ComputeBackend(
    val displayName: String,
    val shortName: String,
    val conditionerUsesNnapi: Boolean,
    val decoderUsesNnapi: Boolean,
    val allowFp16: Boolean,
    val pipelineConditioner: Boolean,
) {
    CPU(
        displayName = "CPU • reliable",
        shortName = "CPU",
        conditionerUsesNnapi = false,
        decoderUsesNnapi = false,
        allowFp16 = false,
        pipelineConditioner = false,
    ),
    NNAPI(
        displayName = "GPU / NPU via NNAPI • automatic",
        shortName = "NNAPI",
        conditionerUsesNnapi = true,
        decoderUsesNnapi = true,
        allowFp16 = false,
        pipelineConditioner = false,
    ),
    NNAPI_FP16(
        displayName = "GPU / NPU via NNAPI + FP16 • speed test",
        shortName = "NNAPI FP16",
        conditionerUsesNnapi = true,
        decoderUsesNnapi = true,
        allowFp16 = true,
        pipelineConditioner = false,
    ),
    PARALLEL_ACCELERATOR_CPU(
        displayName = "Parallel GPU/NPU conditioner + CPU decoder • experimental",
        shortName = "Parallel accel+CPU",
        conditionerUsesNnapi = true,
        decoderUsesNnapi = false,
        allowFp16 = true,
        pipelineConditioner = true,
    ),
}
