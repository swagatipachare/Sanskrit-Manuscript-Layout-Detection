import os
import cv2
import json
import argparse

from pathlib import Path
from ultralytics import YOLO

# ==========================================
# STEP 1: PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = (
    BASE_DIR / "runs" / "detect" /
    "manuscript_layout" / "weights" / "best.pt"
)

# ==========================================
# STEP 2: IMAGE EXTENSIONS
# ==========================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

# ==========================================
# STEP 3: GET INPUT IMAGES
# ==========================================

def get_images(input_path):

    input_path = Path(input_path)

    if input_path.is_file():

        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported image format.")

        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    images = []

    for file in input_path.rglob("*"):

        if file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    if not images:
        raise FileNotFoundError(
            "No images found in input folder."
        )

    return sorted(images)

# ==========================================
# STEP 4: MAIN INFERENCE FUNCTION
# ==========================================

def run_inference(input_path, output_path, weights):

    input_path = Path(input_path)
    output_path = Path(output_path)
    weights = Path(weights)

    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}. "
            "Run train.py first."
        )

    # Load trained YOLO model
    model = YOLO(str(weights))

    print("YOLOv11 model loaded successfully!")

    # Create output directories
    annotated_dir = output_path / "annotated"

    json_dir = output_path / "json"

    annotated_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Get images
    image_files = get_images(input_path)

    all_predictions = []

    # ======================================
    # STEP 5: PROCESS EACH IMAGE
    # ======================================

    for image_path in image_files:

        print("\nProcessing:", image_path.name)

        image = cv2.imread(str(image_path))

        if image is None:

            print("Unable to read image.")
            continue

        height, width = image.shape[:2]

        # YOLO prediction
        results = model.predict(

            source=image,

            conf=0.25,

            imgsz=640,

            device="cpu",

            verbose=False

        )

        result = results[0]

        detections = []

        # ==================================
        # STEP 6: EXTRACT BOUNDING BOXES
        # ==================================

        for box in result.boxes:

            class_id = int(box.cls.item())

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            # Clip coordinates to image boundary
            x1 = max(0, min(x1, width - 1))

            y1 = max(0, min(y1, height - 1))

            x2 = max(0, min(x2, width - 1))

            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = result.names[class_id]

            detections.append({

                "class_id": class_id,

                "class_name": class_name,

                "confidence": round(
                    confidence, 4
                ),

                "bounding_box": [

                    round(x1, 2),

                    round(y1, 2),

                    round(x2, 2),

                    round(y2, 2)

                ]

            })

        # ==================================
        # STEP 7: SAVE ANNOTATED IMAGE
        # ==================================

        annotated_image = result.plot()

        output_image = (
            annotated_dir /
            f"{image_path.stem}_detected.jpg"
        )

        cv2.imwrite(
            str(output_image),
            annotated_image
        )

        # ==================================
        # STEP 8: CREATE JSON METADATA
        # ==================================

        prediction = {

            "image_name": image_path.name,

            "image_width": width,

            "image_height": height,

            "total_detections": len(detections),

            "detections": detections

        }

        all_predictions.append(prediction)

        json_file = (
            json_dir /
            f"{image_path.stem}.json"
        )

        with open(
            json_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                prediction,
                file,
                indent=4
            )

        print(
            "Detected regions:",
            len(detections)
        )

    # ======================================
    # STEP 9: SAVE COMBINED JSON
    # ======================================

    combined_json = (
        output_path / "predictions.json"
    )

    with open(
        combined_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_predictions,
            file,
            indent=4
        )

    print("\n================================")

    print("INFERENCE COMPLETED")

    print("Total images:", len(all_predictions))

    print("Annotated images:", annotated_dir)

    print("JSON metadata:", combined_json)

    print("================================")


# ==========================================
# STEP 10: COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Manuscript Layout Region Detection"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or folder path"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder path"
    )

    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Trained YOLO model path"
    )

    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        weights=args.weights
    )