from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("yolov8n-seg.pt")

def lane_detection_instance_segmentation(image_path):
    """Mendeteksi jalur menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result.jpg")

# Contoh penggunaan
lane_detection_instance_segmentation("dataset.jpg")
