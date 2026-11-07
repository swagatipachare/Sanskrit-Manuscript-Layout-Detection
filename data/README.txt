# Sanskrit Manuscript Layout Dataset (Augmented Sample)

## Contents
- 80 JPG images: 1 original sample + 79 augmented variants
- YOLO-format bounding-box labels
- data.yaml with five classes
- Train/validation/test folders

## Classes
0: header
1: footer
2: main_text
3: side_text
4: filler

## Important provenance and limitations
All images in this package are derived from ONE user-provided Google Images sample.
They are not 80 independent historical manuscript pages and must not be represented as
an independently collected 80-page historical dataset.

The bounding boxes are approximate manual region annotations based on the visible
sample. In particular, filler regions group visible red margin rules and red separators.
Review/correct every annotation in Roboflow or another annotation tool before final
assignment submission.

The train/validation/test split is provided for YOLO pipeline compatibility only.
Because all images originate from the same source page, the split has source leakage
and cannot provide a valid estimate of generalization to unseen manuscripts.
For a defensible evaluation, collect distinct manuscript pages and keep all augmented
versions of each source page within the same split.

## Augmentation
Mild rotation, scaling/translation, brightness/contrast, low-level noise, occasional
slight blur, and small saturation changes. Geometric transforms were also applied to
the bounding boxes.

## Source
Original source image: supplied by the user in this conversation.
