# Robot-Keypoints

Keypoint RCNN based robot joints detection. Original implementation is from [Mask R-CNN](https://arxiv.org/abs/1703.06870):

![Example 0](./images/visualized_result_000045.png)

[Test image source]: Deep Robot-to-Camera Extrinsics for Articulated Manipulators ([DREAM](https://github.com/NVlabs/DREAM))


## Table of Contents

- [Download Process](#download-process)
- [How to Run](#how-to-run)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Annotation format](#annotation-format)

---

## Download Process

    git clone https://github.com/kidpaul94/Robot_Keypoints.git
    pip3 install opencv-python albumentations torch torchvision

## How to Run

### Training:

    python3 custom_train.py 

### Evaluation:

> **Note**
`custom_train.py`, `eval.py` have several arguments to pass. Run the `--help` command to see more information.

    python3 eval.py 

## Annotation format

    {"bboxes": [[x_c, y_c, w, h]], "keypoints": [[[x1, y1, v1], [x2, y2, v2], [x3, y3, v3], ... , [x7, y7, v7]]]}
