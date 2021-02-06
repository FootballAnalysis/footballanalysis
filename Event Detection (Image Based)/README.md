# Event Dataset (Image Based)   -- (Repository Under Construction)

## General Info

Using this module, you can process a full video of a soccer match, obtaining all of the events that occurred in the match. The input could be a soccer match with any time duration, while the output, after processing and distinguishing events, is a tagged video, in which the events occurred at any time will be highlighted by a tag on the video.
This service currently could distinguish seven events as follows (more events will be added as soon as possible):


- Penalty Kick
- Free Kick
- Corner Kick
- Red Card
- Yellow Card
- Tackle
- Substitute


<br/>
<div align="center">
<img src="Images/Algorithm.jpg" alt="Algorithm"  width="928" height="162.5" >  
<figcaption>Figure 1. Proposed algorithm </figcaption>

</div>

<br/>

This module has been implemented using Neural Networks. The proposed algorithm of this network is presented in fig (1). 
our work has some improvements and advantages as below:
- using of the [EfficientNet](https://arxiv.org/abs/1905.11946) architecture for image classification
- using the Fine-grained architecture and solving the problems related to yellow and red card classification
- Improving the recognition of No Highlights images and event images using 3 methods
1. using of VAE.
2. Setting a Threshold in the Classification Network.
3. Using 3 image classes in EfficientNet to ensure that the event images are not mistakenly assigned to the classes related to these events.

- Using our new [Soccer event dataset](https://gitlab.com/footballanalysis/FootballAnalysis/-/tree/master/Datasets/Soccer%20Event%20Dataset%20(Image)).


## Demo Application

If you want to train networks from scratch, Please Skip this section and go to the Training section.

To deploy the demo, run the following commands:
- Download Code Folder
- change directory to Event Detection Module Repository.
- Install Python dependencies: `pip install -r requirements.txt`
- Put your football into "video/input/" directory.
- Run the demo: `python event_demo.py `

The results will be saved in "video/output/" directory.

## Training

To train networks from scratch, run the following commands:
- Download Code Folder
- Install Python3
- Install Python dependencies: `pip install -r requirements.txt`
- Run the code: `python main.py `


## Some Results

- sample of event detection :

<br/>
<div align="center">
<img src="Images/img_event2.jpg" alt="Event Detection (Image based)"  width="200" height="306" >  
<figcaption>Figure 2. sample input of event detection</figcaption>

</div>

<br/>
<div align="center">
<img src="Images/img_event3.png" alt="Event Detection (Video based)"  width="266" height="205" >  
<figcaption>Figure 3. sample output of event detection</figcaption>

</div>


## Citation

When using this code in your research, please cite us:

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={}
}
```

