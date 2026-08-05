#!/usr/bin/env python3
"""Idempotently wire the generated K8/V8 implementation into the existing Kotlin app."""
from __future__ import annotations

from pathlib import Path


ENGINE = Path("app/src/main/java/dev/cardrhyme/muscriptormobile/MuScriptorEngine.kt")
ACTIVITY = Path("app/src/main/java/dev/cardrhyme/muscriptormobile/MainActivity.kt")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_engine() -> None:
    text = ENGINE.read_text()
    if "private val customOpLibraryPath: String? = null" in text:
        print("MuScriptorEngine.kt already patched")
        return

    text = replace_once(
        text,
        """    private val cachePrecision: CachePrecision = CachePrecision.FP32,\n    private val requestedBackend: ComputeBackend = ComputeBackend.CPU,\n    private val threadCount: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 8),\n""",
        """    private val cachePrecision: CachePrecision = CachePrecision.FP32,\n    private val requestedBackend: ComputeBackend = ComputeBackend.CPU,\n    private val customOpLibraryPath: String? = null,\n    private val threadCount: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 8),\n""",
        "engine constructor",
    )
    text = replace_once(
        text,
        """    private val cache = SharedCache(environment, maxCacheLength, cachePrecision)\n    private val emptyCondition = floatTensor(FloatArray(0), longArrayOf(1, 0, MODEL_DIM.toLong()))\n""",
        """    private val cache = if (cachePrecision == CachePrecision.K8V8_NATIVE) {\n        null\n    } else {\n        SharedCache(environment, maxCacheLength, cachePrecision)\n    }\n    private val k8v8Decoder = if (cachePrecision == CachePrecision.K8V8_NATIVE) {\n        K8V8Decoder(\n            environment = environment,\n            modelDir = modelDir,\n            maxCacheLength = maxCacheLength,\n            customOpLibraryPath = requireNotNull(customOpLibraryPath) {\n                \"K8/V8 mode requires the packaged native custom-op library path\"\n            },\n            threadCount = threadCount,\n        )\n    } else {\n        null\n    }\n    private val emptyCondition = floatTensor(FloatArray(0), longArrayOf(1, 0, MODEL_DIM.toLong()))\n""",
        "engine cache initialization",
    )
    text = replace_once(
        text,
        """        val condition = prepared.tensor\n        val conditionLength = prepared.length\n\n        val firstIds = LongArray(prompt.size + 1)\n""",
        """        val condition = prepared.tensor\n        val conditionLength = prepared.length\n        val nativeK8V8 = k8v8Decoder\n        if (nativeK8V8 != null) {\n            val result = nativeK8V8.run(\n                bootstrapSession = decoder,\n                condition = condition,\n                conditionLength = conditionLength,\n                prompt = prompt,\n                onToken = onToken,\n            )\n            return ChunkResult(\n                tokensNotStreamed = 0,\n                lastTokenMillis = result.lastTokenMillis,\n                hitCacheLimit = result.hitCacheLimit,\n            )\n        }\n        val sharedCache = checkNotNull(cache)\n\n        val firstIds = LongArray(prompt.size + 1)\n""",
        "K8V8 run branch",
    )
    text = replace_once(text, "                putAll(cache.inputs)\n", "                putAll(sharedCache.inputs)\n", "normal cache inputs")
    text = replace_once(
        text,
        "            val result = decoder.run(inputs, linkedSetOf(\"logits\"), cache.outputs)\n",
        "            val result = decoder.run(inputs, linkedSetOf(\"logits\"), sharedCache.outputs)\n",
        "normal cache outputs",
    )
    text = replace_once(
        text,
        """            CachePrecision.Storage.PACKED_UINT4 -> {\n""",
        """            CachePrecision.Storage.K8V8 -> {\n                error(\"K8/V8 uses its dedicated data + scale cache implementation\")\n            }\n            CachePrecision.Storage.PACKED_UINT4 -> {\n""",
        "shared cache exhaustive storage",
    )
    text = replace_once(
        text,
        """        emptyCondition.close()\n        cache.close()\n        decoder.close()\n""",
        """        emptyCondition.close()\n        k8v8Decoder?.close()\n        cache?.close()\n        decoder.close()\n""",
        "engine close",
    )
    text = replace_once(
        text,
        """        append(String.format(Locale.US, \"%.1f MiB\", cachePrecision.actualMemoryMiB(maxCacheLength)))\n    }\n""",
        """        append(String.format(Locale.US, \"%.1f MiB\", cachePrecision.actualMemoryMiB(maxCacheLength)))\n        if (cachePrecision == CachePrecision.K8V8_NATIVE) {\n            append(\" • FP16 prefill + CPU fused INT8 incremental attention\")\n        }\n    }\n""",
        "engine backend status",
    )
    ENGINE.write_text(text)
    print("patched MuScriptorEngine.kt")


def patch_activity() -> None:
    text = ACTIVITY.read_text()
    if "customOpLibraryPath = if (cachePrecision == CachePrecision.K8V8_NATIVE)" in text:
        print("MainActivity.kt already patched")
        return

    text = replace_once(
        text,
        "INT4 model • adaptive FP32 / FP16 / BF16 / INT8 / INT4 KV cache • 1 s overlap",
        "INT4 model • native FP16 + K8/V8 cache paths • 1 s overlap",
        "header description",
    )
    text = replace_once(
        text,
        """                        cachePrecision = cachePrecision,\n                        requestedBackend = requestedBackend,\n                    ).use { engine ->\n""",
        """                        cachePrecision = cachePrecision,\n                        requestedBackend = requestedBackend,\n                        customOpLibraryPath = if (cachePrecision == CachePrecision.K8V8_NATIVE) {\n                            K8V8Native.libraryPath(this@MainActivity)\n                        } else {\n                            null\n                        },\n                    ).use { engine ->\n""",
        "engine custom library argument",
    )
    text = replace_once(
        text,
        """            \"%d MiB budget → %d positions · %.1f MiB persistent KV · ~%d generation positions. %s%s Adapter workspace is additional.\",\n""",
        """            \"%d MiB budget → %d positions · %.1f MiB persistent KV · ~%d generation positions. %s%s %s\",\n""",
        "cache summary format",
    )
    text = replace_once(
        text,
        """            precision.qualityNote,\n            capNote,\n        )\n""",
        """            precision.qualityNote,\n            capNote,\n            when {\n                precision == CachePrecision.K8V8_NATIVE -> \"A temporary 96 MiB FP16 prefill cache is additional.\"\n                precision.usesFullCacheBoundaryConversion -> \"Adapter workspace is additional.\"\n                else -> \"\"\n            },\n        )\n""",
        "cache summary runtime note",
    )
    ACTIVITY.write_text(text)
    print("patched MainActivity.kt")


def main() -> None:
    patch_engine()
    patch_activity()


if __name__ == "__main__":
    main()
