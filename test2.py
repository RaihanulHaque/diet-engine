import os
import cv2
import numpy as np
from inference import ModelDetector

class FoodPredictor(ModelDetector):
    def __init__(self, model_path="yolo-Weights/diet_engine_all_best_4.pt"):
        super().__init__(model_path)

    def get_food_size_and_calorie(self, name, area_ratio):
        """
        Estimate the size and calorie content of the detected food item.
        """
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
        else:
            food_size = None
            calorie = 0
        return food_size, calorie

    def process_frame(self, frame) -> np.ndarray:
        """
        Process a single frame, detect food items, estimate their sizes and calories,
        and annotate the frame with the results.
        """
        img_height, img_width, _ = frame.shape
        original_area = img_width * img_height
        total_calories = 0

        # Perform detection using the YOLO model
        results = self.model(frame, device='mps')[0]
        detections = results.boxes

        for box in detections:
            x1, y1, x2, y2 = [round(coord) for coord in box.xyxy[0].tolist()]
            area = (x2 - x1) * (y2 - y1)
            class_id = box.cls[0].item()
            probability = round(box.conf[0].item(), 2)

            # Skip low-confidence detections
            if probability < 0.3:
                continue

            name = results.names[class_id]
            area_ratio = original_area / area

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            try:
                if probability > 0.6:
                    # Get food size and calorie information
                    food_size, calorie = self.get_food_size_and_calorie(name, area_ratio)
                    total_calories += calorie

                    # Create label with food name, probability, size, and calorie info
                    label = f"{name} - {probability} - {food_size}" if food_size else f"{name} - {probability}"
                    label += f" - {calorie} cal"

                    # Add label to the frame
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error processing food item: {e}")

        # Display total calories on the frame
        cv2.putText(frame, f"Total Calories: {total_calories}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, total_calories

    def process_images_in_folder(self, input_folder, output_folder=None):
        """
        Process all images in the input folder, perform inference, and save/display the results.
        If output_folder is provided, save the annotated images there.
        """
        # Ensure the output folder exists if specified
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get all image files in the input folder
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Process the image
            annotated_image, total_calories = self.process_frame(image)

            # Display the annotated image
            cv2.imshow("Food Detection", annotated_image)
            cv2.waitKey(0)  # Wait for a key press to move to the next image

            # Save the annotated image if an output folder is provided
            if output_folder:
                output_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_path, annotated_image)
                print(f"Saved annotated image to: {output_path}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    predictor = FoodPredictor()

    # Specify the input folder containing images and optionally an output folder
    input_folder = "images"
    # output_folder = "path/to/output/folder"  # Set to None if you don't want to save images

    # Process all images in the input folder
    predictor.process_images_in_folder(input_folder)