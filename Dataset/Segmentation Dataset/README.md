# Segmentation dataset

## Dataset Description

- In this project, a segmentation dataset has been prepared specifically for football. The purpose of collecting segmentation datasets is to provide basic data for training MASK R-CNN and DEEP LAB networks, which are part of segmentation networks. These networks are used to identify players and goals on the football field. After identifying the players and the goalS, this module is used in the virtual advertising section.
- Generally 600 photos were tagged by members with the tools available on hasty.ai, which was reduced to 500 photos after cleaning.


<div align="center">
<img src="Images/segmentationdataset.png" alt="Segmentation Dataset"  width="500" height="320" >  
<figcaption>Figure 1. samples of segmentation dataset</figcaption>

</div>
<br/>

<div align="center">
<img src="Images/segmentationdataset2.png" alt="Segmentation Dataset 2"  width="500" height="280" >  
<figcaption>Figure 2. samples of segmentation dataset</figcaption>

</div>

<br/>

Starter code for the dataset can be found [here](https://gitlab.com/footballanalysis/FootballAnalysis/-/tree/master/Virtual%20advertising). In addition to training code, you will also find python scripts for evaluating standard metrics for comparisons between models.



## Dataset Structure

Below is a overview of the segmentation dataset.


```bash
SD___Dataset.zip
    |
    |── Annotation          (Annotation Files) 
    |     ├── train
    |      |
    |     └── validation
    |
    |── Masks               (Mask images) 
    |    ├── train
    |    |
    |    └── validation
    |
    └── Images               (Original images)      
         ├── train
         |
         └── validation                 
```

## How to download dataset


You can download the dataset from [here](https://#)

## Citation

When using this dataset in your research, please cite us:

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={}
}
```
