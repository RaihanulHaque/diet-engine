from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

class VideoProcessor:
    def __init__(self, source_video) -> None:
        self.source_video = source_video
        # self.model = YOLO("yolo-Weights/diet_engine_all_best_4.pt")
        self.model = YOLO("yolov8n.pt")
        self.box_annotator = sv.BoxAnnotator()


    def show_video(self):
        cap = cv2.VideoCapture(self.source_video)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.flip(frame, 1)
            frame, total_calories = self.process_frame(frame)
            cv2.putText(frame, f"Total Calories: {total_calories}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


    def predict(self, frame: np.ndarray) -> np.ndarray:
        return self.model.predict(frame)[0]
    
    def get_food_size_and_calorie(self, name, area_ratio):
        if name == 'apple':
            food_size = "Big" #if area_ratio > 4.0 and area_ratio < 6.0 else "Small"
            calorie = 95 #if food_size == "Big" else 78
        elif name == 'banana':
            food_size = "Small" #if area_ratio > 0 and area_ratio < 6.0 else "Small"
            calorie = 95 #if food_size == "Big" else 95 121 hobe
        elif name == 'orange':
            food_size = "Big" #if area_ratio > 4.0 and area_ratio < 6.0 else "Small"
            calorie = 62 #if food_size == "Big" else 47
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

        result = sv.Detections.from_ultralytics(self.predict(frame))
        # print(result.xyxy) # print the bounding boxes as list
        labels = []
        for in_result, xyxy in zip(result, result.xyxy):
            x1, y1, x2, y2 = [round(x) for x in xyxy]
            area = (x2 - x1) * (y2 - y1)
            area_ratio = original_area / area
            class_id = in_result[3]
            probability = round(in_result[2], 2)
            name = self.model.names[class_id]
            if name == 'person':
                continue
            labels.append(f"{name} - {probability:0.2f}")
            try:
                if probability > 0.6:
                    food_size, calorie = self.get_food_size_and_calorie(name, area_ratio)
                    total_calories += calorie
                    if name == 'person':
                        continue
                    if food_size:
                        labels[-1] += f" - {food_size} - {calorie} cal"
                        # cv2.putText(frame, f"{name} - {probability:0.2f} - {food_size} - {calorie} cal", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        labels[-1] += f" - {calorie} cal"
                        # cv2.putText(frame, f"{name} - {probability:0.2f} - {calorie} cal", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # cv2.putText(frame, f"{name} - {probability}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except:
                pass
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.putText(frame, f"{name} - {probability}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        # labels = [
        #     f"{self.model.names[in_result[3]]} - {in_result[2]:.2f}"
        #     for in_result in result
        # ]

        frame = self.box_annotator.annotate(scene=frame, detections=result, labels=labels)

        
        # result = self.predict(frame)
        # for box in result.boxes:
        #     food_size = None
        #     x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        #     area = (x2 - x1) * (y2 - y1)
        #     class_id = box.cls[0].item()
        #     probability = round(box.conf[0].item(), 2)
        #     if probability < 0.3:
        #         continue

        #     name = result.names[class_id]
        #     area_ratio = original_area / area
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #     try:
        #         if probability > 0.6:
        #             food_size, calorie = self.get_food_size_and_calorie(name, area_ratio)
        #             total_calories += calorie
        #             if food_size:
        #                 cv2.putText(frame, f"{name} - {probability} - {food_size} - {calorie} cal", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #             else:
        #                 cv2.putText(frame, f"{name} - {probability} - {calorie} cal", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #             # cv2.putText(frame, f"{name} - {probability}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #     except:
        #         pass
        
        return frame, total_calories


if __name__ == "__main__":
    # source_video = "resources/walking_people.mp4"
    # source_video = 0
    source_video = 'http://192.168.0.112:8080/video'
    processor = VideoProcessor(source_video)
    processor.show_video()