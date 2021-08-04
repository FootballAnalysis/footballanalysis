# Bird's Eye View

## General Info

Understanding the 3D layout of a scene from a single perspective image is one of the fundamental problems in computer vision.
The position of the camera next to the playground changes according to where the focus of the game is at that moment. However, with current technology and computer vision, we can modify this viewing experience from a fully 2D perspective to a near 3D experience that includes a top view and this can be useful for many applications that need player information to evaluate them. We specifically tried to draw a bird's eye view for soccer games. 


- We go through three steps to get the Bird's Eye View:

   1. Object Detection

   2. Tracking

   3. Perspective Transformation
   
   4. Color Detection 


## Yolov5 for Object Detection
Tracking an object and drawing a bird's eye view of that object requires the installation of bounding boxes around that object in the image. For this purpose, Object Detection is used. It identifies and indicates a location of objects in bounding boxes in an image.
Therefore, The first step is that the `player` and `ball` must be identified in the camera view.

We trained our [custom dataset](https://github.com/FootballAnalysis/footballanalysis/tree/main/Dataset/Object%20Detection%20Dataset) on [Yolov5](https://github.com/ultralytics/yolov5) for both the player and ball classes.
- If you want to train our dataset from scratch or want to train on your dataset, please refer to [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.


## Deep-Sort for Tracking

The most popular and one of the most widely used, elegant object tracking framework is [Deep SORT](https://arxiv.org/pdf/1703.07402.pdf). Tracking has many uses, including determining the average speed of players, getting their heatmaps, and many more.
We used it here to track players and assign unique IDs to them.
 
- If you want to train Deep Sort from scratch, please refer to [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.
- The implementation of this part is taken from [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).

## K-means for Color Detection
The structure of this section is drawn in Figure 1, wherein the K-means algorithm has been used to detect the color of the player's jersey. We have MxN pixels for each picture, each of which is made up of three components: Red, Green, and Blue. We'll take these MxN pixels as data points and use K-means to cluster them. The output of K-means is the clusters of the colors which has more similarity with each other. 
Here are two classes to fit K-means. Due to the green color of the grass in the background of the image, the most likely color to be detected will be green. For this purpose, the second color will be considered.
For better display, the detected color distance is calculated with the colors in the palette formed by the base colors, and the base color with the shortest distance from the detected color is selected for display.


![Color](https://user-images.githubusercontent.com/61879630/125197281-38154800-e272-11eb-936c-c3c47182890e.PNG)
                                                        
 <p align="center">
	  Figure 1. The Structure of Color Detection
</p>

## Bird's Eye View

To draw a bird's eye view, we need a hemography matrix, and here we obtain this matrix using the Perspective transformation Module described [here](https://github.com/FootballAnalysis/footballanalysis/tree/main/Perspective%20Transformation).
We then obtain the new coordinates of the detected objects center using the hemographic matrix obtained by the Perspective transformation Module.

- To transform the location of a point you can use the following transformation:

<p align="center">
    <img src="/Images/Transformation-Formula.jpg" width = 584px height = 60px><br/>
</p>



- Where `(x, y)` is the original point, and `M` is our perspective matrix.

Finally, we draw new points on a screen that has a top view of the football field.

You can see the output result below.


<p align="center">
    <img src="/Images/Bird.gif" width = 618px height = 346px><br/>
	 Figure 2. sample output of Bird's Eye View
</p>




- The two teams are separated by blue and white according to the color of their jersey.
- The ball is shown in yellow.
- The refree is shown in pink according to the color of his jersey.


## Requirements

Please run the following code to install the Requirements.

`pip install -r requirements.txt`


## Preparation and Run the code:

1. Download the models from [here](https://docs.google.com/uc?export=download&id=1EaBmCzl4xnuebfoQnxU1xQgNmBy7mWi2) and place them under `weights/`.
2. Test on video and see the demo results using the command mentioned in the next line:
```bash
$ python3 track.py --source test_video.mp4 [--view] [--save]
```
3. If you use the `--save` argument, the output will be saved in the `inference/output` folder.
