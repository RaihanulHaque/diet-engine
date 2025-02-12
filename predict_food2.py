import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import cv2
import os


def get_prediction(image):
    img_height, img_width, _ = image.shape
    original_area = img_width * img_height
    model = YOLO("yolo-Weights/diet_engine_all_best_4.pt")
    results = model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        food_size = None
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        area = (x2 - x1) * (y2 - y1)
        class_id = box.cls[0].item()
        probability = round(box.conf[0].item(), 2)
        if probability < 0.1:
            continue

        name = result.names[class_id]
        area_ratio = original_area / area

        if name in ['apple', 'banana', 'orange']:
            if name == 'apple':
                food_size = "Big" if area_ratio > 4.0 and area_ratio < 6.0 else "Small"
            elif name == 'banana':
                food_size = "Big" if area_ratio > 0 and area_ratio < 6.0 else "Small"
            elif name == 'orange':
                food_size = "Big" if area_ratio > 4.0 and area_ratio < 6.0 else "Small"

        prediction = {
            'name': name,
            'probability': probability,
            'area_ratio': original_area / area,
            'food_size': food_size,
        }
        output.append(prediction)
    return output


def create_pie_chart(class_counts):
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
    ax.axis('equal')
    ax.set_title('Class Distribution of Detected Items')

    # Save chart as an image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def combine_visuals(total_images, detected_images, class_counts, all_predictions):
    detection_rate = (detected_images / total_images) * 100 if total_images > 0 else 0

    # Create pie chart
    pie_chart = create_pie_chart(class_counts)

    # Create a figure to combine visuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Visualization 1: Benchmark Summary
    axes[0].text(0.5, 0.5, f"Detection Rate: {detection_rate:.2f}%\n"
                           f"Total Images: {total_images}\n"
                           f"Detected Images: {detected_images}",
                 ha='center', va='center', fontsize=14, bbox=dict(boxstyle="round", facecolor="wheat"))
    axes[0].axis('off')

    # Visualization 2: Pie Chart
    axes[1].imshow(pie_chart)
    axes[1].axis('off')

    # Visualization 3: Example Predictions
    sample_output = "\n".join([f"{pred['name']} ({pred['food_size']}) - {pred['probability']*100:.1f}%"
                               for pred in all_predictions[:10]])
    axes[2].text(0.5, 0.5, f"Sample Predictions:\n{sample_output}",
                 ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor="lightblue"))
    axes[2].axis('off')

    # Save the final combined image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def benchmark_model_with_visuals(image_folder):
    total_images = 0
    detected_images = 0
    all_predictions = []

    class_counts = {'apple': 0, 'banana': 0, 'orange': 0, 'other': 0}

    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            predictions = get_prediction(image)
            if predictions:
                detected_images += 1
                all_predictions.extend(predictions)
                for pred in predictions:
                    name = pred['name']
                    class_counts[name] = class_counts.get(name, 0) + 1

    # Generate visuals
    final_image = combine_visuals(total_images, detected_images, class_counts, all_predictions)

    # Save or display the combined image
    final_image.save('benchmark_visuals.png')
    final_image.show()


# Example usage
image_folder = 'images'
benchmark_model_with_visuals(image_folder)
