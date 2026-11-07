import os
from pathlib import Path
import yaml
from ultralytics import YOLO

# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

DATA_YAML = DATA_DIR / "data.yaml"

MODEL_PATH = BASE_DIR / "yolo11n.pt"

# ==========================================
# STEP 2: CHECK DATASET
# ==========================================

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"data.yaml not found: {DATA_YAML}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

# Check image and label folders
for split in ["train", "val", "test"]:

    image_folder = DATA_DIR / "images" / split
    label_folder = DATA_DIR / "labels" / split

    if not image_folder.exists():
        raise FileNotFoundError(
            f"Image folder missing: {image_folder}"
        )

    if not label_folder.exists():
        raise FileNotFoundError(
            f"Label folder missing: {label_folder}"
        )

    images = list(image_folder.glob("*"))
    labels = list(label_folder.glob("*.txt"))

    print(f"\n{split.upper()} DATASET")
    print("Images:", len(images))
    print("Labels:", len(labels))

# ==========================================
# STEP 3: FIX YAML PATH FOR WINDOWS
# ==========================================

with open(DATA_YAML, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Use the actual Windows dataset directory
config["path"] = str(DATA_DIR.resolve())

runtime_yaml = DATA_DIR / "data_runtime.yaml"

with open(runtime_yaml, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        config,
        file,
        sort_keys=False
    )

print("\nDataset YAML:", runtime_yaml)

# ==========================================
# STEP 4: LOAD YOLOv11 MODEL
# ==========================================

model = YOLO(str(MODEL_PATH))

print("\nYOLOv11 model loaded successfully!")

# ==========================================
# STEP 5: TRAIN MODEL
# ==========================================

results = model.train(

    data=str(runtime_yaml),

    epochs=80,

    imgsz=640,

    batch=4,

    device="cpu",

    workers=2,

    patience=20,

    project=str(BASE_DIR / "runs" / "detect"),

    name="manuscript_layout",

    save=True,

    plots=True,

    verbose=True

)

print("\nTraining completed successfully!")

print(
    "Best model saved at:",
    BASE_DIR / "runs" / "detect" /
    "manuscript_layout" / "weights" / "best.pt"
)