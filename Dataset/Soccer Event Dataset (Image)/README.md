# Soccer Event Dataset (Image)

## Dataset Description


In the present project, two image datasets of a football match were collected: 
1. the Soccer Event (SEV) dataset covering the football match events
2. the Test Event dataset used to assess the proposed architecture 

The aforementioned datasets were collected in two ways:

1. Web crawling and collection of images related to events
2. Watching the videos of UCL and European league football matches to extract the relevant frames.


The SEV dataset covered 7 football match events and 3 scenes from the football field. The seven main events include:
- **Corner Kick**
- **Penalty Kick**
- **Free Kick**
- **Red Card**
- **Yellow Card**
- **Tackle**
- **substitute**

The scenes from the football field included:
- **Left Penalty Area**
- **Right Penalty Area**
- **Center Circle**


<div align="center">
<img src="Images/ImageDataset.jpg" alt="Image Dataset"  width="500" height="320" >  
<figcaption>Figure 1. Samples of soccer event dataset</figcaption>

</div>
<br/>

The Test Event dataset is used to test the event detection network. This dataset consists of 3 classes:

1. Images of selected events in the SEV database (a total of 200 images were extracted from 7 events)
2. Football match images that don’t fall within the category of the 7 events.
3. Other Images (Nature,Car,....)

<br/>
<div align="center">
<img src="Images/ImageDataset2.jpg" alt="Image Dataset 2"  width="500" height="280" >  
<figcaption>Figure 2. Samples of test event dataset</figcaption>

</div>

<br/>
<br/>

Starter code for the dataset can be found [here](https://gitlab.com/footballanalysis/FootballAnalysis/-/tree/master/Event%20Detection%20(Image%20Based)). In addition to training code, you will also find python scripts for evaluating standard metrics for comparisons between models.

# Dataset Statistics


- The Soccer event dataset consists of 60000 images from 10 different events, each of which includes 6000 images.
- In each category, 5000 images are used as the training data, 500 images are used as validation data and 500 images are used as test data

<br/>

<div align="center">
<img src="Images/ImageDataset_Statistics.JPG" alt="ImageDataset_Statistics"  width="394" height="287" >  
<figcaption>Table 1. Soccer Event Dataset</figcaption>

</div>

<br/>

- The Test Event dataset includes 4,200 images falling within 3 classes. This dataset is described in detail in table (2) 
<br/>

<div align="center">
<img src="Images/ImageDataset_Statistics2.JPG" alt="ImageDataset_Statistics2"  width="300" height="192" >  
<figcaption>Table 2. Test Event Dataset</figcaption>

</div>

## Dataset Structure

Below is a overview of the soccer event dataset.

```bash
SEV___Dataset.zip
    |
    |── Train (directories of Events names)
    |     ├── Penaly Kick
    |     |
    |     |── Free Kick
    |     |
    |     └── .....   
    |── Validation (directories of Events names)
    |     ├── Penaly Kick
    |     |
    |     |── Free Kick
    |     |
    |     └── .....   
    └── Test (directories of Events names)
          ├── Penaly Kick
          |
          |── Free Kick
          |
          └── .....                          
```

## How to download dataset


You can download the dataset from [here](https://#)


# Citation

When using this dataset in your research, please cite us:

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={}
}
```

