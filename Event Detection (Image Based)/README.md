# Event Dataset (Image Based)

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


<p align="center">
    <img src="./Images/Algorithm.jpg" width = 928px height = 163px><br/>
	Figure 1. Proposed algorithm 
</p>


<br/>

This module has been implemented using Neural Networks. The proposed algorithm of this network is presented in fig (1). 
our work has some improvements and advantages as below:
- using the new architecture for image classification
- using the Fine-grained architecture and solving the problems related to yellow and red card classification
- Improving the recognition of No Highlights images and event images using 3 methods
1. using of VAE.
2. Setting a Threshold in the Classification Network.
3. Using 3 image classes in image classification module to ensure that the event images are not mistakenly assigned to the classes related to these events.

- Using our new [Soccer event dataset](https://github.com/FootballAnalysis/footballanalysis/tree/main/Dataset/Soccer%20Event%20Dataset%20(Image)).




## Some Results

- sample of event detection :

<p align="center">
    <img src="./Images/img_event2.jpg" width = 200px height = 306px><br/>
	Figure 2. sample input of event detection
</p>

<br/>

<p align="center">
    <img src="./Images/img_event3.png" width = 266px height = 205px><br/>
	Figure 3. sample output of event detection
</p>




## Citation

When using this code in your research, please cite us:

```
@article{karimi2021soccer,
  title={Soccer Event Detection Using Deep Learning},
  author={Karimi, Ali and Toosi, Ramin and Akhaee, Mohammad Ali},
  journal={arXiv preprint arXiv:2102.04331},
  year={2021}
}
```

