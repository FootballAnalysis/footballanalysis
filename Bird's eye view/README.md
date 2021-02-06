# Bird's Eye View

## General Info

Understanding the 3D layout of a scene from a single perspective image is one of the fundamental problems in computer vision.
The position of the camera next to the playground changes according to where the focus of the game is at that moment. However, with current technology and computer vision, we can modify this viewing experience from a fully 2D perspective to a near 3D experience that includes a top view and this can be useful for many applications that need player information to evaluate them. We specifically tried to draw a bird's eye view for soccer games. 


- We go through three steps to get the Bird's Eye View:

   1. Object Detection

   2. Tracking

   3. Perspective Transform


## Yolov5 for Object Detection
Tracking an object and drawing a bird's eye view of that object requires the installation of bounding boxes around that object in the image. For this purpose, Object Detection is used. It identifies and indicates a location of objects in bounding boxes in an image.
Therefore, The first step is that the `ball` and `players` must be identified in the camera view.

We trained our [custom dataset](https://gitlab.com/footballanalysis/FootballAnalysis/-/tree/master/Datasets/Object%20Detection%20Dataset) on [Yolov5](https://github.com/ultralytics/yolov5) for both the player and ball classes.
- If you want to train our dataset from scratch or want to train on your dataset, please refer to [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.

You can see the results in below.

|              |   Percision    |    Recall     |
|    :---:     |     :---:      |    :---:      |
|   YOLOv5s    |     0.881      |    0.916      |
|   YOLOv5m    |    soon        |    soon       |
|   YOLOv5l    |    soon        |    soon       |
|   YOLOv5x    |    soon        |    soon       |

## Deep-Sort for Tracking With PyTorch 

The most popular and one of the most widely used, elegant object tracking framework is [Deep SORT](https://arxiv.org/pdf/1703.07402.pdf).Tracking has many uses, including determining the average speed of players, getting their heatmaps, and many more.
We used it here to track players and assign unique IDs to them.
 
- If you want to train Deep Sort from scratch, please refer to [this site](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more information.

## Bird's Eye View

To draw a bird's eye view, we need a hemography matrix, and here we obtain this matrix using the Perspective transformation Module described [here](https://gitlab.com/footballanalysis/FootballAnalysis/-/tree/master/Perspective%20Transformation).
We then obtain the new coordinates of the detected objects center using the hemographic matrix obtained by the Perspective transformation Module.

- To transform the location of a point you can use the following transformation:

<div align="center">
<img src="Images/Transformation-Formula.jpg" alt="Bird's eye view"  width="584" height="60" >  
</div>

- Where `(x, y)` is the original point, and `M` is our perspective matrix.

Finally, we draw new points on a screen that has a top view of the football field.

You can see the output result below.

<div align="center">
<img src="Images/Bird.gif" alt="Bird's eye view"  width="618" height="346" >  
<figcaption> Figure 1. sample output of Bird's Eye View </figcaption>
</div>



- The two teams are separated by blue and white according to the color of their jersey.
- The ball is shown in yellow (If the ball is not detected, it uses the previously detected coordinates).
- The refrees are shown in pink according to the color of their jersey.
- **To identify the color of the players' and referee's jersey, you can use the Yolo object detection to extract the image of each player from the test video and then train a simple classification on it.**

## Requirements

Please run the following code to install the Requirements.

`pip install -U -r requirements.txt`

## Pre-trained Models

- Yolov5 Pre-trained Models on our Custom Dataset:
   - [YOLOv5s Model]()
   - [YOLOv5m Model]()
   - [YOLOv5l Model]()
   - [YOLOv5x Model]()
- Deep Sort Pre-trained Model :
   - [Deep_Sort Model]()

## Preparation and Run the code:

1. Download the Yolov5 pre-trained model and place the downlaoded `.pt` file under `yolov5/weights/`.
2. Download the Deep Sort pre-trained model and place the downlaoded `ckpt.t7` file under `deep_sort/deep/checkpoint/`.
6. Test on video and see the demo results using the command mentioned in the next line:
```bash
$ python3 track.py --source test_video.mp4
```
