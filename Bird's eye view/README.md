# Bird's Eye View

## General Info

Understanding the 3D layout of a scene from a single perspective image is one of the fundamental problems in computer vision. The position of the camera next to the playground varies based on the focus point of the camera. However, with the advancement of technology and computer vision, we can modify the viewing experience from a fully 2D perspective to a near 3D experience included the top view that provides beneficial information to assess soccer players. In the following, it has been attempted to explain how to draw the bird's eye view of a soccer game.

- We go through three steps to get the Bird's Eye View:

   1. Object Detection

   2. Tracking

   3. Color Detection 

   4. Perspective Transformation
   

## Yolov5 for Object Detection
Tracking an object and drawing the bird's eye view requires the installation of bounding boxes around the specified object in the image. Therefore, we must first identify the ball and player classes.

We trained our [custom dataset](https://github.com/FootballAnalysis/footballanalysis/tree/main/Dataset/Object%20Detection%20Dataset) on [Yolov5](https://github.com/ultralytics/yolov5) for both the `player` and `ball` classes.
- If you want to train our dataset from scratch or want to train on your dataset, please refer to [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.


## Deep-Sort for Tracking

One of the most widely used and elegant object tracking frameworks is [Deep SORT](https://arxiv.org/pdf/1703.07402.pdf). Tracking has many applications, including determining the average speed of players, getting their heatmaps, and many more. Here, Deep-sort has been utilized to track players and assign unique IDs to them.
 
- If you want to train Deep Sort from scratch, please refer to [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.
- The implementation of this part is taken from [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).

## K-means for Color Detection
The structure of this section is drawn in Figure 1, wherein the K-means algorithm has been used to recognize the players' jersey color.

We have MxN pixels for each image, each of which consists of three components: Red, Green, and Blue. We will take these MxN pixels as data points and use K-means to cluster them. The outputs of K-means are the clusters of data points (RGB colors) that have repeated more. Here are two classes to fit K-means. Due to the green color of the grass, the most likely color to be detected will be green. For this purpose, the second color will be considered.

For better display, the detected color distance is calculated with the base colors in the palette. Lastly, the base color with the shortest distance from the detected color will be selected for display.

![Color](https://user-images.githubusercontent.com/61879630/125197281-38154800-e272-11eb-936c-c3c47182890e.PNG)
                                                        
 <p align="center">
	  Figure 1. The Structure of Color Detection
</p>

## Bird's Eye View

To draw the bird's eye view, we need a hemography matrix, and here we obtain it matrix using the Perspective transformation Module described [here](https://github.com/FootballAnalysis/footballanalysis/tree/main/Perspective%20Transformation).

We introduce the center of the lower side of the detected bounding box as the detected object coordinate.
We then attain the new coordinates of the detected object using the homographic matrix obtained by the perspective transformation module.

- The following transformation is utilized to transform the specified coordinate:

<p align="center">
    <img src="/Images/Transformation-Formula.jpg" width = 584px height = 60px><br/>
</p>

- Where `(x, y)` is the original point, and `M` is the perspective matrix.

Finally, we draw new points on the top view image.

You can see the output result below.

<p align="center">
    <img src="/Images/Bird.gif" width = 618px height = 346px><br/>
	 Figure 2. sample output of Bird's Eye View
</p>


- The two teams are separated by blue and white according to the color of their jersey.
- The ball is colored yellow.
- The refree is colored pink according to the color of his jersey.


## Requirements

Please run the following code to install the Requirements.

`pip install -r requirements.txt`


## Preparation and Run the code:

1. Download the models from [here](https://docs.google.com/uc?export=download&id=1EaBmCzl4xnuebfoQnxU1xQgNmBy7mWi2) and place them under `weights/`.
2. Test on your sample video and see the demo results using the command mentioned in the next line:
```bash
$ python3 main.py --source test_video.mp4 [--view] [--save]
```
3. If you use the `--save` argument, the output will be saved in the `inference/output` folder.
