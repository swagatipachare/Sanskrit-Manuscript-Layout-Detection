from pathlib import Path
import yaml
from ultralytics import YOLO

# ==========================================
# STEP 1: PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODEL_PATH = (
    BASE_DIR / "runs" / "detect" /
    "manuscript_layout" / "weights" / "best.pt"
)

DATA_YAML = DATA_DIR / "data.yaml"

# ==========================================
# STEP 2: CHECK FILES
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train.py first."
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

# ==========================================
# STEP 3: CREATE WINDOWS YAML
# ==========================================

with open(DATA_YAML, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

config["path"] = str(DATA_DIR.resolve())

runtime_yaml = DATA_DIR / "data_runtime.yaml"

with open(runtime_yaml, "w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, sort_keys=False)

# ==========================================
# STEP 4: LOAD TRAINED MODEL
# ==========================================

model = YOLO(str(MODEL_PATH))

print("Trained YOLOv11 model loaded!")

# ==========================================
# STEP 5: MODEL EVALUATION
# ==========================================

metrics = model.val(

    data=str(runtime_yaml),

    split="test",

    imgsz=640,

    batch=4,

    device="cpu",

    plots=True,

    verbose=True

)

# ==========================================
# STEP 6: DISPLAY RESULTS
# ==========================================

print("\n========== MODEL EVALUATION ==========")

print(
    "Precision:",
    round(metrics.box.mp, 4)
)

print(
    "Recall:",
    round(metrics.box.mr, 4)
)

print(
    "mAP@50:",
    round(metrics.box.map50, 4)
)

print(
    "mAP@50-95:",
    round(metrics.box.map, 4)
)

print("\nClass-wise mAP@50:")

for class_id, score in zip(
    metrics.box.ap_class_index,
    metrics.box.ap50
):

    print(
        f"Class {int(class_id)}:",
        round(float(score), 4)
    )

print("\nEvaluation completed!")