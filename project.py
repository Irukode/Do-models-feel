from comet_ml import Experiment
import torch
import torch.nn as nn
import numpy as np
from models import GPT, BERT
from data import preprocessGPT, preprocessBERT

run_GPT = True #TODO: change this depending on whether we are running BERT or GPT

gpt_hyperparams = {
    "batch_size": 64,
    "num_epochs": 5,
    "learning_rate": 0.004,
    "num_heads": 8,
    "num_layers": 6,
    "d_model": 512,
    "seq_len": 192
 }

bert_hyperparams = {
    "batch_size": 64,
    "num_epochs": 5 * 100/15, #adjusts for only training on 15% of data
    "learning_rate": 0.004,
    "num_heads": 8,
    "num_layers": 6,
    "d_model": 512,
    "seq_len": 192
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = Experiment(project_name="final_project", log_code=True)
if (run_GPT):
    experiment.log_parameters(gpt_hyperparams)
else:
    experiment.log_parameters(bert_hyperparams)

if __name__ == "__main__":
    if run_GPT:
        print("loading data for GPT model")
        vocab_size, train_loader, fine_tuning_loader, validation_loader = preprocessGPT()
    else:
        print("loading data for BERT model")
        train_loader = 0
        fine_tuning_loader = 0
        vocab_size = 0
        validation_loader = 0

    #create model
    if run_GPT:
        model = GPT(device=device, seq_len=gpt_hyperparams["seq_len"], num_words=vocab_size,
                    d_model=gpt_hyperparams["d_model"], h= gpt_hyperparams["num_heads"],
                    n=gpt_hyperparams["num_layers"])
    else:
        model = BERT(device=device, seq_len=gpt_hyperparams["seq_len"], num_words=vocab_size,
                    d_model=gpt_hyperparams["d_model"], h=gpt_hyperparams["num_heads"],
                    n=gpt_hyperparams["num_layers"])

    #train model
    print("training model ...")

    #fine tune model
    print("fine tuning model ...")

    #validate model
    print("validating mode ...")