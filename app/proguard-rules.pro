# Keep TensorFlow Lite GPU delegate classes (loaded reflectively).
-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.gpu.**
