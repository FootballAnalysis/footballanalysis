# KeywordsSpotting-EfficientNet-A0
EfficientNet-Absolute Zero for Continuous Speech Keyword Spotting

This is a PyTorch implementation of some popular CNN models architecture like [Deep Residual Models](https://arxiv.org/abs/1710.10361), [Convolutional Neural Networks for Keyword Spotting](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html), and our proposed architecture based on [EfficientNet](https://arxiv.org/abs/1905.11946). all models are trained on our new Persian Keyword Spotting Dataset that you can download from [Football Keywords Dataset](https://drive.google.com/file/d/1m0CoqVzneGVxfTx-uGpAXdZvPNgH9gjS/view?usp=sharing). For more details, please check out our paper [EfficientNet-Absolute Zero for Continuous Speech Keyword Spotting
](https://arxiv.org/abs/2012.15695).

This repository is based on [Honk-Repository](https://github.com/castorini/honk-models). Honk models can be used to identify simple commands (e.g., "stop" and "go") that trained on [Speech Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html). but our work has some improvements and advantages as below:
1. We used the modified state of the art image classification architecture, `efficientNet`, as a based model.
2. Improve performance in "continuous speech" mode by our Proposed `continuous speech synthesis method`.
3. Improve robustness against noises in real samples by using various noises like bubble, stadium, ... .
4. Better generalization by using [SpecAugment](https://arxiv.org/abs/1904.08779).
5. Using our new Persian Keywords Spotting Dataset that helped us to use this project in real scenarios and projects.

## Demo Application
Use the instructions below to run the demo application (shown in the above video) yourself!
Currently, PyTorch has official support for only Linux and OS X. Thus, Windows users will not be able to run this demo easily.

To deploy the demo, run the following commands:
- change directory to KSM Repository.
- If you do not have PyTorch, please see [the website](http://pytorch.org).
- Install Python dependencies: `pip install -r requirements.txt`
- Start the PyTorch server: `python .`
- Run the demo: `python -m utils.speech_demo_tk`

If you need to adjust options, like turning off CUDA or change trained model file or ... ,  please edit `config.json`.

## Pre trained  Models
As soon as possible we release [KSM trained-models](https://#) in our repository. there are several pre-trained models for PyTorch.

## Contact Us
Feel free to contact us for any further information via below channels.

### Amirmohhammad Rostami:
- email: [*amirmohammadrostami@yahoo.com*](amirmohammadrostami@yahoo.com), [*a.m.rostami@aut.ac.ir*](a.m.rostami@aut.ac.ir)
- linkdin: [*amirmohammadrostami*](https://www.linkedin.com/in/amirmohammadrostami/)
- homepage: [*amirmohammadrostami*](https://ce.aut.ac.ir/~amirmohammadrostami/)


### Ali Karimi
- email: [*aliiikarimi@ut.ac.ir*](aliiikarimi@ut.ac.ir)

### Mohammad Ali Akhaee
- email: [*akhaee@ut.ac.ir*](akhaee@ut.ac.ir)
- homepage: [*Mohammad Ali Akhaee*](https://ece.ut.ac.ir/en/~akhaee)



## Cite us

@article{rostami2022keyword,
  title={Keyword spotting in continuous speech using convolutional neural network},
  author={Rostami, Amir Mohammad and Karimi, Ali and Akhaee, Mohammad Ali},
  journal={Speech Communication},
  year={2022},
  publisher={Elsevier}
}
