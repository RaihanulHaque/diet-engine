from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

class VideoProcessor:
    def __init__(self, source_video) -> None:
        self.source_video = source_video
        # self.model = YOLO("yolo-Weights/diet_engine_all_best_4.pt")
        self.model = YOLO("yolo-Weights/yolo11m.pt")
        self.box_annotator = sv.BoxAnnotator()

    def show_video(self):
        cap = cv2.VideoCapture(self.source_video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame, total_calories = self.process_frame(frame)
            cv2.putText(frame, f"Total Calories: {total_calories}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def predict(self, frame: np.ndarray):
        # return self.model.predict(frame)[0]
        return self.model(frame, device='mps')[0]

    def get_food_size_and_calorie(self, name, area_ratio):
        if name == 'apple':
            food_size = "Big" if 4.0 < area_ratio < 6.0 else "Small"
            calorie = 95 if food_size == "Big" else 78
        elif name == 'banana':
            food_size = "Big" if 0 < area_ratio < 6.0 else "Small"
            calorie = 121 if food_size == "Big" else 95
        elif name == 'orange':
            food_size = "Big" if 4.0 < area_ratio < 6.0 else "Small"
            calorie = 62 if food_size == "Big" else 47
        elif name == 'milk':
            food_size = None
            calorie = 150
        elif name == 'bread':
            food_size = None
            calorie = 80
        elif name == 'fried_egg':
            food_size = None
            calorie = 90
        return food_size, calorie

    def process_frame(self, frame) -> np.ndarray:
        img_height, img_width, _ = frame.shape
        original_area = img_width * img_height
        total_calories = 0
        result = self.predict(frame)

        # Uncomment the following lines to print the bounding box coordinates as a list
        # print(result.xyxy)

        for box in result.boxes:
            x1, y1, x2, y2 = [round(coord) for coord in box.xyxy[0].tolist()]
            area = (x2 - x1) * (y2 - y1)
            class_id = box.cls[0].item()
            probability = round(box.conf[0].item(), 2)

            if probability < 0.3:
                continue

            name = result.names[class_id]
            area_ratio = original_area / area
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            try:
                if probability > 0.6:
                    food_size, calorie = self.get_food_size_and_calorie(name, area_ratio)
                    total_calories += calorie
                    label = f"{name} - {probability} - {food_size}" if food_size else f"{name} - {probability}"
                    label += f" - {calorie} cal"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except:
                pass

        return frame, total_calories

if __name__ == "__main__":
    # source_video = "resources/walking_people.mp4"
    source_video = 0
    processor = VideoProcessor(source_video)
    processor.show_video()
