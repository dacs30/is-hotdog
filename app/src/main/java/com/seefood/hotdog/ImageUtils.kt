package com.seefood.hotdog

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

/** Camera-frame conversion helpers. */
object ImageUtils {

    /**
     * Convert a CameraX [ImageProxy] to an upright [Bitmap].
     *
     * The analysis use case is configured for RGBA_8888 output, so [ImageProxy.toBitmap]
     * gives us a bitmap directly; we then rotate it by the frame's reported rotation so
     * the model sees the scene the right way up.
     */
    fun toUprightBitmap(image: ImageProxy): Bitmap {
        val bitmap = image.toBitmap()
        val rotation = image.imageInfo.rotationDegrees
        if (rotation == 0) return bitmap

        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
        if (rotated != bitmap) bitmap.recycle()
        return rotated
    }
}
