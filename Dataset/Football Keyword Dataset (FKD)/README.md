# Football Keywords Dataset (FKD)

### General Info

- In order to prepare this dataset, an attempt has been made to collect various data in terms of `age`, `gender`, `emotions`, and `tone of expression` of the speaker, as well as the recording device and environment, so that the system being trained can be adequately generalized.

- This dataset has been collected in `Persian` and by Persian speakers to solve the problem of lack of keyword spotting dataset in Persian. On the other hand, the keywords used in this dataset are similar and very close to the same words in English or other languages, and there are only differences in pronunciation and accent. Words like goal, corner, penalty, and hand in most languages have the same meaning and usage. Therefore, the use of these data can be used for generalization of other datasets and languages with transfer learning, which is an advantage for this dataset.

- The list of words is shown in below `table`. As it turns out, the majority of important events that take place in football, which can be identified through the voice of the reporter, have been taken into account in this dataset. On the other hand, during the collection of data, seven types of expression `Normal`, `Emphasized`, `Upset`, `Surprised`, `Emotional`, `Fast` and `Stretched`'have been tried to be collected from the speakers so that the comprehensiveness of the dataset is very appropriate. This point, along with the variety of recording systems and ambient noise, makes this Dataset general and utilizable in real applications.

|  # |   Keyword   | # of Train Utterances | # of Test Utterances | # of CSS Utterances | # of Unique Speaker |
|:--:|:-----------:|:---------------------:|:--------------------:|:-------------------:|:-------------------:|
|  1 |    Corner   |          1322         |          300         |          22         |         387         |
|  2 |     Foul    |          1369         |          300         |          3          |         366         |
|  3 |  Free kick  |          1555         |          300         |          80         |         376         |
|  4 |     Goal    |          1379         |          300         |          53         |         392         |
|  5 |  Goalposts  |          1376         |          300         |         182         |         369         |
|  6 |     Hand    |          1621         |          300         |          53         |         414         |
|  7 |  Laying off |          1330         |          300         |          72         |         429         |
|  8 |    Mulct    |          1577         |          300         |          12         |         410         |
|  9 |    Notice   |          1311         |          300         |          39         |         432         |
| 10 |   Offside   |          1294         |          300         |          56         |         469         |
| 11 |     Out     |          1355         |          300         |          56         |         405         |
| 12 |   Penalty   |          1420         |          300         |         114         |         395         |
| 13 |   Red card  |          1238         |          300         |          30         |         448         |
| 14 |    Strike   |          1396         |          300         |          93         |         380         |
| 15 |  Substitute |          1334         |          300         |          81         |         419         |
| 16 |    Tackle   |          1245         |          300         |          34         |          426        |
| 17 |   Throw in  |          1352         |          300         |          73         |          444        |
| 18 | Yellow card |          1307         |          300         |          23         |         475         |
| 19 |   Unknown   |           -           |          300         |          20         |          -          |
| 20 |   Silence   |           -           |          300         |          20         |          -          |
| - |    Total    |         24811         |         6000         |         1116        |        ≈ 1800       |

- Dataset is divided into two parts `Train` and `Test`. To do this, for each class, 300 samples are provided for Test and the rest for Train. In addition, for the selection of Test samples, we considered condition `at least 100 samples for each category that speakers are not present in the Train data`.

- To evaluate the models in Continuous Speech, there are 500 `2-second` continuous speech samples in `Continuous Speech (CS) folder`.

- This dataset can also be used for Speaker Verification (SV), given that there is an average of 10 samples from each speaker.

- Two telegram bots [GozareshgarBot](https://telegram.me/Bot_Gozareshgar) and [CrowdBot](https://telegram.me/VoiceReaction_1_bot) has been created to collect audio files as crowdsourcing with OGG Vorbis format. For more details about how to clean and use for continuous speech, please wait for our paper [EfficientNet-Absolute Zero for Continuous Speech Keyword Spotting
](https://arxiv.org/abs/2012.15695).

- Data set splited to test and train part



### Citation

When using this dataset in your research, please cite us:


```
@article{rostami2022keyword,
  title={Keyword spotting in continuous speech using convolutional neural network},
  author={Rostami, Amir Mohammad and Karimi, Ali and Akhaee, Mohammad Ali},
  journal={Speech Communication},
  year={2022},
  publisher={Elsevier}
}
```
