# Prototypical Functions of Artifacts
This is the code for our paper [Learning Prototypical Functions for Physical Artifacts](https://aclanthology.org/2021.acl-long.540.pdf) (ACL 2021).

## Testing Environment
This project is built on python==3.7.6, torch=1.4.0, transformers==2.9.0.

## Selected Frames and Human Annotations
We selected 42 frames from FrameNet 1.7 (plus "None of above") to represent actions that are common functions of human-made physical artifacts. The frame names and definitions can be found in "frame_defs.tsv".

Our gold standard data set contains 938 artifacts that are each paired with one frame that represents its most prototypical use. They can be found in "gold_annotations.tsv".

## Data Format
For train and dev, input csv files should contain three columns: artifact, definition, label. <br>
For test/prediction, input csv files should contain two columns: artifact, definition. <br>

artifact: the target word or phrase <br>
definition: the dictionary definition of the artifact <br>
label: an integer indicating the correct frame

## Use Our Pre-trained Models
Download the pre-trained models at: [here](https://drive.google.com/file/d/1XLPgft8EtsoCj-v0A_M7ee7Z98NKTkJb/view?usp=sharing).

The file you would like to predict will be "data/artifact_function/test.csv" by default, but you can also use a specific file for prediction using the "--test_file" argument.

Put the extracted "model_artifact_function" folder under the "pretrained_models/" directory, and run predict.sh.

It will generate the predicted file in the model folder (e.g., "pretrained_models/model_artifact_function/predicted_test.csv").


An input file for prediction should be like:
| artifact  | definition |
| --------- | ---------- |
| paintbrush|a brush used as an applicator (to apply paint) |
| luggage   | cases used to carry belongings when traveling |
|shoestring |a lace used for fastening shoes |

The output file for prediction will be like:
| artifact  | definition | predicted frame |
| --------- | ---------- | --------------- |
| paintbrush|a brush used as an applicator (to apply paint) | Create_representation |
| luggage   | cases used to carry belongings when traveling | Containing  |
|shoestring |a lace used for fastening shoes | Closure |

This pre-trained model is trained on all our gold standard data, so it should not be evaluated on our test set. To replicate the evaluation results in the paper, you should train from scratch as instructed below.


## Train from Scratch
We set aside 20% (188) of the data as a development set and used 80% (750) as the test set. We evaluated the models by performing 5-fold cross validation on the test set.

Run train.sh.

To get evaluation results, run eval.py.


## Contact and Reference
For questions and issues, please contact `tianyu@cs.utah.edu`. Our paper can be cited as:
```
@inproceedings{jiang-riloff-2021-learning,
title="{Learning Prototypical Functions for Physical Artifacts}",
author={Jiang, Tianyu and Riloff, Ellen},
booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)},
year={2021}
}
```
