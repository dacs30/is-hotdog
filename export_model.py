from ultralytics import YOLO
m = YOLO("yolov8n.pt")          # downloads ~6MB nano model
m.export(format="tflite", imgsz=640)   # -> yolov8n_saved_model/yolov8n_float32.tflite
print("EXPORT_DONE")
