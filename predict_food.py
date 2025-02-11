from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

class VideoProcessor:
    def __init__(self, source_video) -> None:
        self.source_video = source_video
        self.model = YOLO("yolo11n.pt")
        self.box_annotator = sv.BoxAnnotator(
            # thickness=2,
            # text_thickness=1,
            # text_scale= 0.5,
        )


    def show_video(self):
        cap = cv2.VideoCapture(self.source_video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


    def predict(self, frame: np.ndarray) -> np.ndarray:
        return self.model.predict(frame)[0]


    def process_frame(self, frame) -> np.ndarray:
        result = sv.Detections.from_ultralytics(self.predict(frame))

        labels = [
            f"{self.model.names[in_result[3]]} - {in_result[2]:.2f}"
            for in_result in result
        ]

        frame = self.box_annotator.annotate(scene=frame, detections=result, labels=labels)

        # for box in result.boxes:
        #     # box.draw(frame)
        #     x1, y1, x2, y2 = [
        #         round(x) for x in box.xyxy[0].tolist()
        #     ]
        #     class_name = result.names[0]
        #     probability = round(box.conf[0].item(), 2)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #     try:
        #         if probability > 0.6:
        #             cv2.putText(frame, f"{class_name} - {probability}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #     except:
        #         pass
        
        return frame


if __name__ == "__main__":
    source_video = "0"
    processor = VideoProcessor(1)
    processor.show_video()