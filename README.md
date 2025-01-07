# Robot-Keypoints
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

Keypoint RCNN based robot joints detection. Original implementation is from [Mask R-CNN](https://arxiv.org/abs/1703.06870):

![Example 0](./images/visualized_result_000045.png)

[Test image source]: Deep Robot-to-Camera Extrinsics for Articulated Manipulators ([DREAM](https://github.com/NVlabs/DREAM))


## Table of Contents

- [Download Process](#download-process)
- [How to Run](#how-to-run)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
- [Annotation format](#annotation-format)

---

## Download Process

> [!NOTE]
This repository has been tested on [Google Colab](https://colab.research.google.com/). It also depends on **opencv-python**, **albumentations**, **torch**, and **torchvision**:

    git clone https://github.com/kidpaul94/Robot_Keypoints.git
    cd Robot_Keypoints/
    pip3 install -r requirements.txt

## How to Run

### Model Training:

> [!NOTE]
`custom_train.py` receives several different arguments. Run the --help command to see everything it receives.

    python3 custom_train.py --help

### Model Evaluation:

> [!NOTE]
`eval.py` receives several different arguments. Run the --help command to see everything it receives.

    python3 eval.py --help

## Annotation format

    {"bboxes": [[x_c, y_c, w, h]], "keypoints": [[[x1, y1, v1], [x2, y2, v2], [x3, y3, v3], ...]]}
