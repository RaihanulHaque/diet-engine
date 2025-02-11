from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt


def get_prediction(image):
    img_height, img_width, _ = image.shape
    original_area = img_width * img_height
    model = YOLO("yolo-Weights/ultimate_diet_engine.pt")
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


def create_summary_image(total_images, detected_images):
    detection_rate = (detected_images / total_images) * 100 if total_images > 0 else 0

    # Create a figure for the summary
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.text(0.5, 0.5, f"Detection Rate: {detection_rate:.2f}%\n"
                       f"Total Images: {total_images}\n"
                       f"Detected Images: {detected_images}",
            ha='center', va='center', fontsize=14, bbox=dict(boxstyle="round", facecolor="wheat"))
    ax.axis('off')

    # Save as an image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_pie_chart(class_counts):
    # Create a pie chart for class distribution
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Class Distribution of Detected Food Items')

    # Save as an image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_predictions_image(all_predictions):
    # Prepare a summary of the top predictions
    sample_output = "\n".join([f"{pred['name']} ({pred['food_size']}) - {pred['probability']*100:.1f}%"
                               for pred in all_predictions[:10]])

    # Create a figure for the predictions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.text(0.5, 0.5, f"Sample Predictions:\n{sample_output}",
            ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor="lightblue"))
    ax.axis('off')

    # Save as an image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def benchmark_model_with_separate_visuals(image_folder):
    total_images = 0
    detected_images = 0
    all_predictions = []
    class_counts = {}

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

    # Generate the three images
    summary_image = create_summary_image(total_images, detected_images)
    pie_chart = create_pie_chart(class_counts)
    predictions_image = create_predictions_image(all_predictions)

    # Save or display the images
    summary_image.save('benchmark_summary.png')
    pie_chart.save('benchmark_pie_chart.png')
    predictions_image.save('benchmark_predictions.png')

    # summary_image.show()
    # pie_chart.show()
    # predictions_image.show()


# Example usage
image_folder = 'images'
benchmark_model_with_separate_visuals(image_folder)
