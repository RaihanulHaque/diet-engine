from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import cv2
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

app = FastAPI()


def get_prediction(image):
    img_height, img_width, _ = image.shape
    original_area = img_width * img_height
    # model = YOLO("yolo-Weights/diet_engine_all_best_4.pt")
    model = YOLO("yolo-Weights/ultimate_diet_engine.pt")
    results = model.predict(image, device=DEVICE)
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
            # Customize size conditions for each fruit
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


@app.get('/')
async def root():
    return {"message": "Hello World"}


@app.post('/image-upload')
async def upload_image(file: UploadFile = File(...)):
    # Check if the file is an image (you can add more comprehensive checks)
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        raise HTTPException(
            status_code=400, detail="Only image files (JPEG, JPG, PNG, GIF) are allowed.")

    # Read the image file using Pillow
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Save the image to a temporary directory or process it as needed
    temp_image_path = f"images/temp.jpg"
    image.save(temp_image_path)
    img = cv2.imread(temp_image_path)
    img_height, img_width, _ = img.shape
    area = img_width * img_height
    output = get_prediction(img)

    # Count the total number of each food and calculate total calories
    total_apple = 0
    total_banana = 0
    total_orange = 0
    total_milk = 0
    total_fried_egg = 0
    total_boiled_egg = 0
    total_bread = 0
    total_calories = 0

    print(output)

    for i in output:
        if i['name'] == 'apple':
            total_apple += 1
            total_calories += 78 if i['food_size'] == 'Small' else 95
            print(f"{i['name']} - {i['food_size']} & {i['area_ratio']}")

        elif i['name'] == 'banana':
            total_banana += 1
            total_calories += 90 if i['food_size'] == 'Small' else 121
            print(f"{i['name']} - {i['food_size']} & {i['area_ratio']}")

        elif i['name'] == 'orange':
            total_orange += 1
            total_calories += 47
            print(f"{i['name']} - {i['food_size']} & {i['area_ratio']}")

        elif i['name'] == 'milk':
            total_milk += 1
            total_calories += 150  # Adjust the calorie value as needed

        elif i['name'] == 'fried_egg':
            total_fried_egg += 1
            total_calories += 90  # Adjust the calorie value as needed

        elif i['name'] == 'boiled_egg':
            total_boiled_egg += 1
            total_calories += 70  # Adjust the calorie value as needed

        elif i['name'] == 'bread':
            total_bread += 1
            total_calories += 80  # Adjust the calorie value as needed

    # Create a dictionary to store information about detected foods
    foods_info = {}

    if total_apple > 0:
        foods_info["Total Apple"] = total_apple

    if total_banana > 0:
        foods_info["Total Banana"] = total_banana

    if total_orange > 0:
        foods_info["Total Orange"] = total_orange

    if total_milk > 0:
        foods_info["Total Milk"] = total_milk

    if total_fried_egg > 0:
        foods_info["Total Fried Egg"] = total_fried_egg

    if total_boiled_egg > 0:
        foods_info["Total Boiled Egg"] = total_boiled_egg

    if total_bread > 0:
        foods_info["Total Bread"] = total_bread

    print(foods_info)
    print(total_calories)

    # You can return a response indicating the image was processed
    return {
        # "message": "Image uploaded and processed successfully.",
        **foods_info,
        "Total Calories": total_calories,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)