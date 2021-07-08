# Object Detection Dataset


## Dataset Description

In the Object Detection Dataset, we have prepared a dataset of bounding boxes for two classes of player and ball from soccer images.The dataset includes images of various soccer scenes and their corresponding labels.

- **1000 images** for both `player` and `ball` classes were annotated by [Hasty.ai](https://hasty.ai). This site has powerful tools such as powerai that uses AI to speed up the process of data creation.

|         | Numbers  | 
| :-----: | :-: |
| **No. Classes** | 2 |
| **No. Images** | 1000 | 
| **No. Boxes** | 14501 | 
| **Avg No. Boxes / No. Images** | 14.501 |



- For each image, there is a text file containing the `class ID`, `Xmin`, `Ymin`, `Xmax`, and `Ymax`, respectively.

- Labels are also available in the data format required for YOLOV training. Please visit [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.

A sample image with its corresponding annotations is given below. **(Figure 1)**

<p align="center">
    <img src="Images/Sample_Image.jpg" width = 800px height = 495px><br/>
	Figure 1. Sample of SOD Dataset
</p>

## Dataset Structure

Below is a overview of the Line Detection dataset.

```bash
SOD_Dataset.zip
    │
    │
    ├──── images
    │
    │
    ├──── annotations
    │         │
    │         │
    │         └ image_1.txt  <class> <Xmin> <Ymin> <Xmax> <Ymax>
    │             
    │ 
    └──── yolov5_annotations
              │
              │
              └ image_1.txt  <class> <x_center> <y_center> <width> <height>

```
- Box coordinates are in normalized xywh format (from 0 - 1) for **yolov5_annotations**. `x_center` and `width` are divided by image width, and `y_center` and `height` by image height.

## How to download dataset
You can download the dataset from [here](https://drive.google.com/uc?export=download&id=1UYEurzB6ZRJUkn75yQJ_yh3YYfPpW4wh).

