package com.seefood.hotdog

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Result of a single frame's detection. */
data class DetectionResult(val isHotdog: Boolean, val confidence: Float)

/**
 * Thin wrapper around a YOLOv8n TFLite interpreter that answers one question:
 * is there a hot dog in this frame?
 *
 * YOLOv8 is pretrained on COCO's 80 classes; class index 52 is "hot dog". We run
 * inference and surface the highest confidence found for that single class.
 */
class HotdogDetector(context: Context) {

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null

    // Input geometry, read from the model so we don't hard-code assumptions.
    private val inputSize: Int
    // Output geometry: YOLOv8 export is [1, 84, 8400] (channels-first) but some
    // exports are [1, 8400, 84]. We detect which at load time.
    private val channelsFirst: Boolean
    private val numChannels: Int
    private val numBoxes: Int

    private val inputBuffer: ByteBuffer
    private val pixels: IntArray

    init {
        val model = FileUtil.loadMappedFile(context, MODEL_FILE)
        val options = Interpreter.Options()

        // Prefer the GPU delegate when the device supports it; fall back to
        // multi-threaded CPU otherwise.
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)
            Log.i(TAG, "Using GPU delegate")
        } else {
            options.numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
            Log.i(TAG, "Using CPU with ${options.numThreads} threads")
        }

        interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0).shape() // [1, 640, 640, 3]
        inputSize = inputShape[1]

        val outputShape = interpreter.getOutputTensor(0).shape() // [1, 84, 8400]
        // The class/box-attribute axis is the smaller of the two non-batch dims.
        channelsFirst = outputShape[1] < outputShape[2]
        numChannels = if (channelsFirst) outputShape[1] else outputShape[2]
        numBoxes = if (channelsFirst) outputShape[2] else outputShape[1]

        inputBuffer = ByteBuffer
            .allocateDirect(inputSize * inputSize * 3 * 4) // float32, RGB
            .apply { order(ByteOrder.nativeOrder()) }
        pixels = IntArray(inputSize * inputSize)

        Log.i(TAG, "Model loaded: input=$inputSize output=${outputShape.joinToString()}")
    }

    fun detect(bitmap: Bitmap): DetectionResult {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        loadInput(resized)
        if (resized != bitmap) resized.recycle()

        val output = Array(1) { Array(numChannels) { FloatArray(numBoxes) } }
        // When channels-last, the interpreter expects [1, numBoxes, numChannels].
        if (channelsFirst) {
            interpreter.run(inputBuffer, output)
        } else {
            val lastOut = Array(1) { Array(numBoxes) { FloatArray(numChannels) } }
            interpreter.run(inputBuffer, lastOut)
            return parseChannelsLast(lastOut[0])
        }
        return parseChannelsFirst(output[0])
    }

    /** output[channel][box] — pick the best "hot dog" score across all boxes. */
    private fun parseChannelsFirst(output: Array<FloatArray>): DetectionResult {
        val classRow = output[BOX_ATTRS + HOT_DOG_CLASS]
        var best = 0f
        for (i in 0 until numBoxes) {
            if (classRow[i] > best) best = classRow[i]
        }
        return DetectionResult(best > CONFIDENCE_THRESHOLD, best)
    }

    /** output[box][channel] — pick the best "hot dog" score across all boxes. */
    private fun parseChannelsLast(output: Array<FloatArray>): DetectionResult {
        var best = 0f
        for (i in 0 until numBoxes) {
            val score = output[i][BOX_ATTRS + HOT_DOG_CLASS]
            if (score > best) best = score
        }
        return DetectionResult(best > CONFIDENCE_THRESHOLD, best)
    }

    /** Resize-normalized RGB into the reusable direct float buffer (NHWC). */
    private fun loadInput(bitmap: Bitmap) {
        inputBuffer.rewind()
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f) // R
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)  // G
            inputBuffer.putFloat((pixel and 0xFF) / 255f)          // B
        }
        inputBuffer.rewind()
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }

    companion object {
        private const val TAG = "HotdogDetector"
        private const val MODEL_FILE = "yolov8n.tflite"
        private const val HOT_DOG_CLASS = 52 // COCO class index for "hot dog"
        private const val BOX_ATTRS = 4      // x, y, w, h precede the class scores
        private const val CONFIDENCE_THRESHOLD = 0.5f
    }
}
