package dev.cardrhyme.muscriptormobile

import android.content.Context
import java.io.File
import java.nio.ByteBuffer

object K8V8Native {
    private const val LIBRARY_NAME = "muscriptor_k8v8"

    init {
        System.loadLibrary(LIBRARY_NAME)
    }

    fun libraryPath(context: Context): String = File(
        context.applicationInfo.nativeLibraryDir,
        System.mapLibraryName(LIBRARY_NAME),
    ).absolutePath

    external fun quantizeFp16Caches(
        sourceBuffers: Array<ByteBuffer>,
        destinationBuffers: Array<ByteBuffer>,
        scaleBuffers: Array<ByteBuffer>,
        sourceLength: Int,
        destinationLength: Int,
        positions: Int,
        heads: Int,
        headSize: Int,
    )
}
