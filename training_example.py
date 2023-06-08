import os
import torch
import torch.nn as nn
import Trainer
import Model

#Put the path to your model here once you have trained one
#then replace the None parameter in trainIters with loadFilename
model = Model.model("AL.csv", "formatted_lines-AL.txt")

if model.model_name != "":
    checkpoint_iter = 10
    loadFilename = os.path.join("data/save", model.model_name, "AL.csv",
                                '{}-{}_{}'.format(model.encoder_n_layers, model.decoder_n_layers, model.hidden_size),
                                '{}_checkpoint.tar'.format(checkpoint_iter))

iterations = 1
print_frequency = 1
save_frequency = 1

Trainer.trainIters(model, "data/save", iterations, print_frequency, save_frequency, 50, None)
