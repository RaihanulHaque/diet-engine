# Diet Engine

This repository contains the code for the Diet Engine project, which includes training and inference scripts for a YOLO-based object detection model. The project is designed to detect and classify various food items in images and videos.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/diet-engine.git
    cd diet-engine
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset Preparation

1. Create a directory for datasets:
    ```sh
    mkdir datasets
    cd datasets
    ```

2. Download the dataset using Roboflow:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("seefood-ugh5r").project("seefood-research")
    dataset = project.version(9).download("yolov11")
    ```

3. Ensure the dataset is in the correct format and location:
    ```sh
    # The dataset should be in the `datasets/SeeFood---Research-9` directory
    ```

## Training

1. Navigate to the project root directory:
    ```sh
    cd /path/to/diet-engine
    ```

2. Run the training script:
    ```sh
    !yolo task=detect mode=train model=yolo11m.pt data=datasets/SeeFood---Research-9/data.yaml epochs=100 imgsz=480 plots=True batch=32
    ```


## Inference

1. Run inference on test images:
    ```sh
    !yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=datasets/SeeFood---Research-9/test/images save=True
    ```

2. Run inference on a video:
    ```python
    from predict_food import VideoProcessor
    
    processor = VideoProcessor(0)
    processor.show_video()
    ```

## Results

- Training and validation results, including loss curves and metrics, are saved in the `runs/detect/train` directory.
- Inference results are saved in the `runs/detect/predict` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.