package dev.cardrhyme.muscriptormobile

import android.content.Context
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

class ModelDownloader(private val context: Context) {
    data class Progress(val downloaded: Long, val total: Long, val fileName: String)

    private data class ModelFile(
        val name: String,
        val size: Long,
        val sha256: String,
    ) {
        val url: String
            get() = "https://huggingface.co/happyme531/muscriptor-medium-onnx/resolve/main/onnx/w4a32_optimized/$name?download=true"
    }

    private val files = listOf(
        ModelFile(
            "conditioner.onnx",
            6_223_112,
            "c1ea7895ade9760538f42f63110a782bcc1606dee89f2ee434db2251434a3c7d",
        ),
        ModelFile(
            "decoder.onnx",
            79_814,
            "74e1181288fa5205754e9661cbccafc5182d035beadd4412da1ecd00515c3de9",
        ),
        ModelFile(
            "decoder.onnx.data",
            217_354_240,
            "4bc6ac2807854632bacc0ef9b5e9e1871545e573194fe7fc197948730aa3c8ef",
        ),
    )

    val modelDir: File = File(context.filesDir, "models/muscriptor-medium-w4a32")
    private val marker: File get() = File(modelDir, ".verified-v1")
    val totalBytes: Long = files.sumOf { it.size }

    fun isReady(): Boolean = marker.exists() && files.all {
        File(modelDir, it.name).length() == it.size
    }

    suspend fun download(
        onProgress: (Progress) -> Unit,
        onStatus: (String) -> Unit,
    ) {
        modelDir.mkdirs()
        marker.delete()

        var completedBytes = files.sumOf { spec ->
            val target = File(modelDir, spec.name)
            if (target.length() == spec.size && sha256(target) == spec.sha256) spec.size else 0L
        }

        for (spec in files) {
            currentCoroutineContext().ensureActive()
            val target = File(modelDir, spec.name)
            if (target.length() == spec.size && sha256(target) == spec.sha256) {
                onStatus("Verified ${spec.name}")
                onProgress(Progress(completedBytes, totalBytes, spec.name))
                continue
            }
            if (target.exists()) target.delete()
            onStatus("Downloading ${spec.name}")
            downloadOne(spec, target) { fileBytes ->
                onProgress(Progress(completedBytes + fileBytes, totalBytes, spec.name))
            }
            onStatus("Verifying ${spec.name}")
            check(target.length() == spec.size) {
                "${spec.name} has ${target.length()} bytes, expected ${spec.size}"
            }
            check(sha256(target) == spec.sha256) { "SHA-256 mismatch for ${spec.name}" }
            completedBytes += spec.size
            onProgress(Progress(completedBytes, totalBytes, spec.name))
        }

        marker.writeText("happyme531/muscriptor-medium-onnx@7d1a2cc14a335f3bbee445147286f215287ab2a4\n")
        onStatus("Model ready")
    }

    private suspend fun downloadOne(
        spec: ModelFile,
        target: File,
        onBytes: (Long) -> Unit,
    ) {
        val partial = File(modelDir, "${spec.name}.part")
        var existing = partial.length().coerceAtMost(spec.size)
        if (partial.length() > spec.size) {
            partial.delete()
            existing = 0
        }

        var connection = open(spec.url, existing)
        if (existing > 0 && connection.responseCode != HttpURLConnection.HTTP_PARTIAL) {
            connection.disconnect()
            partial.delete()
            existing = 0
            connection = open(spec.url, 0)
        }
        check(connection.responseCode in 200..299) {
            "HTTP ${connection.responseCode} while downloading ${spec.name}"
        }

        BufferedInputStream(connection.inputStream, 1 shl 20).use { input ->
            BufferedOutputStream(FileOutputStream(partial, existing > 0), 1 shl 20).use { output ->
                val buffer = ByteArray(1 shl 20)
                var written = existing
                onBytes(written)
                while (true) {
                    currentCoroutineContext().ensureActive()
                    val count = input.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    written += count
                    onBytes(written)
                }
            }
        }
        connection.disconnect()
        check(partial.renameTo(target)) { "Could not finalize ${spec.name}" }
    }

    private fun open(url: String, offset: Long): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "MuScriptorMobile/0.1")
            if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
            connect()
        }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
