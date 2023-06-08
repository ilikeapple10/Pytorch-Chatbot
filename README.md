# About
This is a library that helps streamline the process of making and training a Pytorch Chatbot from scratch

## Files

## Model
This file contains the model architecture which includes the attention model, batch and hidden size as well as the learning rate and dropout.
This file also contains the two evaluation methods that allow for interactions with the model

## NNetwork
this file contains the attention, encoder and decoder layers for the model

## Trainer
This is the training process the model goes through to learn based on a provided dataset

## Vocabulary
This file is responsible for adding padding around sentence pairs as well as trimming unwanted characters or words before training

## text_extractor
this file converts a dataset into the sentence pairs that are prepared in the above file

## Usage

1. Clone the repo and install requirements

2. in a new file, import os, Trainer and Model

3. in a seperate file, import text_extractor and call the "write_to_file" method. put whatever you want for the datafile but make sure you put the CSV file in as the second parameter

4. in the file with the Trainer and Model files imported, create an instance of the model class and put the paths of the two new files in as parameters

5. Once the file is finished running, you now have a chatbot

## Notes
As you can see, this was hastily made. i will be cleaning this up over time and adding new features down the road. if anything is broken or you want to add something, just make a new issue or pull request