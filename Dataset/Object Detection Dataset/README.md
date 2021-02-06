# Object Detection Dataset


## Dataset Description

In the Object Detection Dataset, we have prepared a dataset of bounding boxes for two classes of player and ball from soccer images.The dataset includes images of various soccer scenes and their corresponding labels.

- **1000 images** for both `ball` and `player` classes were annotated by [Hasty.ai](https://hasty.ai). This site has powerful tools such as powerai that uses AI to speed up the process of data creation.

|         | train  | validation  |
| :-----: | :-: | :-: |
| **No. Class** | 2 | 2 |
| **No. Image** | 800 | 200 |
| **No. Box** | 9643 | 1532 |
| **Avg No. Box / Img** | 9643 | 1532 |



- For each image, there is a text file containing the `class ID`, `Xmin`, `Ymin`, `Xmax`, and `Ymax`, respectively.

- Labels are also available in the data format required for YOLOV training. Please visit [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.

A sample image with its corresponding annotations is given below. **(Figure 1)**.


<div align="center">
<img src="Images/Sample_Image.jpg" alt="Object Detection Dataset"  width="800" height="495" >  
<figcaption> Figure 1. Sample of Object Detection Dataset </figcaption>
</div>


## Dataset Structure

Below is a overview of the Line Detection dataset.

```bash
SOD___Dataset.zip
    |
    └── dataset
          ├──── Main
          |       ├── annotations
          │       │       ├── train
          |       |       |     | 
          |       |       |     └ image_1.txt  <class> <Xmin> <Ymin> <Xmax> <Ymax>
          |       |       |
          │       │       └── validation
          │       └── images
          │             
          │ 
          └──── Yolov_format
                  ├── annotations
                  │       ├── train
                  |       |     | 
                  |       |     └ image_1.txt  <class> <x_center> <y_center> <width> <height>
                  |       |
                  │       └── validation
                  └── images       
```
- Box coordinates are in normalized xywh format (from 0 - 1) for **Yolov_format** dataset. `x_center` and `width` are divided by image width, and `y_center` and `height` by image height.
