# Sanskrit Manuscript Layout Detection Dataset
### YOLO-Based Object Detection Dataset

## 1. Project Overview

This dataset is prepared for Sanskrit manuscript layout analysis using computer vision and deep learning techniques.

The objective is to identify and localize different structural regions of a manuscript page using a YOLO-based object detection model.

Each image is accompanied by YOLO-format bounding-box annotations to support the detection and classification of manuscript layout regions.

## 2. Dataset Information

| Attribute | Details |
|---|---|
| Dataset name | Sanskrit Manuscript Layout Dataset |
| Image format | JPG |
| Annotation format | YOLO |
| Number of classes | 5 |
| Detection framework | YOLO |
| Dataset type | Manuscript layout detection dataset |

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

The bounding boxes represent approximate manual region annotations based on the visible manuscript content. Review and correct the annotations in Roboflow or another annotation tool before final model training or assignment submission.

## 6. Dataset Provenance and Limitations

The dataset is intended for Sanskrit manuscript layout detection and preliminary experimentation.

The annotations are approximate and require manual verification, particularly for the `filler` class, which includes visible red margin rules and red separators.

The quality and diversity of the dataset can influence model performance. Therefore, evaluation results should be interpreted carefully and should not automatically be considered evidence of generalization to all types of historical Sanskrit manuscripts.

## 7. Recommendations for Model Training

For more reliable object-detection results:

1. Collect additional distinct Sanskrit manuscript pages.
2. Annotate each manuscript image accurately.
3. Review all bounding boxes and class assignments.
4. Divide the original manuscript pages into training, validation, and testing sets.
5. Apply suitable data augmentation to the training set when required.
6. Keep related images from the same source page within the same dataset split.
7. Evaluate the trained model using independent, previously unseen manuscript pages.

## 8. Conclusion

This dataset provides a structured foundation for experimenting with Sanskrit manuscript layout detection using YOLO-based object detection.

It supports the identification and localization of important manuscript regions, including headers, footers, main text, side text and decorative elements.

Further improvements in image diversity and annotation accuracy can help develop a more reliable manuscript layout detection system.
