import torch
from ultralytics import YOLO

# Define allowed object classes
TARGET_CLASSES = {"car", "motorcycle", "bus", "truck"}

class YOLOv10Model:
    def __init__(self, model_name="yolov10n"):
        self.model = YOLO(f"{model_name}.pt")  # Load YOLOv10 model

    def detect_objects(self, image, confidence_threshold=0.3):
        results = self.model(image)
        detections = []
        class_names = self.model.names  # Get class names from model

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, score, class_id = box.tolist()
                class_name = class_names.get(int(class_id), "object")  # Convert class ID to name

                # Only keep the required object classes
                if class_name in TARGET_CLASSES and score >= confidence_threshold:
                    detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

        return detections


