# Sanskrit Manuscript Layout Detection Dataset
### Augmented Sample Dataset for YOLO-Based Object Detection

## 1. Project Overview

This dataset is prepared for Sanskrit manuscript layout analysis using computer vision and deep learning techniques.

The objective is to identify and localize different structural regions of a manuscript page using a YOLO-based object detection model.

The dataset contains one original manuscript image and augmented versions created using image-processing techniques. Each image is accompanied by YOLO-format bounding-box annotations.

## 2. Dataset Information

| Attribute | Details |
|---|---|
| Dataset name | Sanskrit Manuscript Layout Dataset |
| Total images | 80 |
| Original images | 1 |
| Augmented images | 79 |
| Image format | JPG |
| Annotation format | YOLO |
| Number of classes | 5 |
| Detection framework | YOLO |
| Dataset type | Augmented sample dataset |

## 3. Object Detection Classes

| Class ID | Class Name | Description |
|---:|---|---|
| 0 | header | Header region of the manuscript |
| 1 | footer | Footer region of the manuscript |
| 2 | main_text | Primary manuscript text region |
| 3 | side_text | Text appearing along the side margins |
| 4 | filler | Visible decorative red margin rules and separators |

## 4. Dataset Structure

```text
Sanskrit_Manuscript_Dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Each image has a corresponding YOLO annotation file.

## 5. Annotation Format

The dataset uses YOLO-format bounding-box annotations. Each annotation represents one object and follows this format:

```text
class_id x_center y_center width height
```

Coordinates are normalized between 0 and 1 relative to the image dimensions.

The bounding boxes are approximate manual region annotations based on the visible manuscript sample. Review and correct every annotation in Roboflow or another annotation tool before final model training or assignment submission.

## 6. Data Augmentation Techniques

The original manuscript image was augmented using:

- **Rotation:** Small-angle rotation to introduce orientation variation.
- **Scaling and translation:** Slight changes in image scale and position.
- **Brightness adjustment:** Variation in image brightness.
- **Contrast adjustment:** Changes in contrast to simulate different image conditions.
- **Noise addition:** Low-level noise to introduce visual variation.
- **Gaussian blur:** Occasional slight blur.
- **Saturation changes:** Small changes in color saturation.

Geometric transformations were also applied to the corresponding bounding boxes to maintain alignment with transformed images.

## 7. Dataset Provenance and Limitations

**Important:** All 80 images in this dataset are derived from ONE user-provided Google Images sample.

The dataset consists of:
- 1 original manuscript image.
- 79 augmented versions of that original image.

These 80 images must not be represented as 80 independently collected historical manuscript pages.

The train, validation, and test folders are provided for YOLO pipeline compatibility. However, because all images originate from the same source page, the split has source leakage.

Consequently, evaluation results from this dataset cannot provide a reliable estimate of model generalization to unseen Sanskrit manuscripts.

The annotations are approximate and require manual verification, particularly for the `filler` class, which includes visible red margin rules and red separators.

## 8. Recommendations for Model Training

For more reliable object-detection results:

1. Collect additional distinct Sanskrit manuscript pages.
2. Annotate each original manuscript image accurately.
3. Review all bounding boxes and class assignments.
4. Split the original pages into training, validation, and testing sets before augmentation.
5. Apply augmentation only to the training set.
6. Keep all augmented versions of a source page within the same split.
7. Evaluate the trained model using independent, previously unseen manuscript pages.

## 9. Source

**Original source image:** One Sanskrit manuscript image supplied by the user in this conversation.

**Dataset preparation:** Image augmentation and approximate YOLO-format region annotations.

## 10. Conclusion

This augmented sample dataset provides an initial dataset structure for experimenting with Sanskrit manuscript layout detection using YOLO-based object detection.

It is suitable for pipeline development and preliminary experimentation, but it should not be considered a representative historical manuscript dataset or used to claim generalization performance without additional independent manuscript images and verified annotations.

