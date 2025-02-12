import cv2
import supervision as sv
from ultralytics import YOLO

class ModelDetector:
    def __init__(self, model_path="yolo-Weights/ultimate_diet_engine.pt"):
        self.model = YOLO(model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    
    def detect_and_annotate(self, image):
        results = self.model(image, device='mps')[0]
        detections = sv.Detections.from_ultralytics(results)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]

        annotated_image = self.box_annotator.annotate(scene=image, detections=detections)
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_image
    
    def process_image(self, image_path):
        image = cv2.imread(image_path)
        annotated_image = self.detect_and_annotate(image)
        cv2.imshow("YOLO Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_webcam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = self.detect_and_annotate(frame)
            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ModelDetector()
    
    # Uncomment the required mode
    detector.process_image("images/boiled1.jpg")
    # detector.process_webcam()
