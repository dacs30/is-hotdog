package com.seefood.hotdog

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.seefood.hotdog.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: HotdogDetector
    private lateinit var analysisExecutor: ExecutorService

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else {
                Toast.makeText(this, R.string.camera_permission_denied, Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detector = HotdogDetector(this)
        analysisExecutor = Executors.newSingleThreadExecutor()

        if (hasCameraPermission()) startCamera()
        else requestPermission.launch(Manifest.permission.CAMERA)
    }

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyze) }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Runs on [analysisExecutor]; converts the frame, detects, then updates the UI. */
    private fun analyze(image: ImageProxy) {
        try {
            val bitmap = ImageUtils.toUprightBitmap(image)
            val result = detector.detect(bitmap)
            bitmap.recycle()
            runOnUiThread { render(result) }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
        } finally {
            image.close()
        }
    }

    private fun render(result: DetectionResult) {
        if (result.isHotdog) {
            binding.resultLabel.setText(R.string.hotdog)
            binding.resultLabel.setBackgroundColor(ContextCompat.getColor(this, R.color.hotdog_green))
        } else {
            binding.resultLabel.setText(R.string.not_hotdog)
            binding.resultLabel.setBackgroundColor(ContextCompat.getColor(this, R.color.not_hotdog_red))
        }
        binding.confidenceLabel.text =
            getString(R.string.confidence_fmt, result.confidence * 100f)
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        detector.close()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
