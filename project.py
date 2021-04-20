from comet_ml import Experiment
import torch
import torch.nn as nn
import numpy as np
from models import GPT, BERT
from data import preprocess_GPT, preprocessBERT

run_GPT = True #TODO: toggle this depending on whether we are running BERT or GPT

gpt_hyperparams = {
    "batch_size": 16,
    "num_epochs": 1,
    "learning_rate": 0.004,
    "num_heads": 4,
    "num_layers": 2,
    "d_model": 256,
    "seq_len": 192
 }

bert_hyperparams = {
    "batch_size": 16,
    "num_epochs": 1 * 100/15, #adjusts for only training on 15% of data
    "learning_rate": 0.004,
    "num_heads": 4,
    "num_layers": 2,
    "d_model": 256,
    "seq_len": 192
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = Experiment(project_name="final_project", log_code=True)
if (run_GPT):
    experiment.log_parameters(gpt_hyperparams)
else:
    experiment.log_parameters(bert_hyperparams)


def train_GPT(model, train_loader, hyperparams):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    model = model.train()
    with experiment.train():
        for epoch_i in range(hyperparams["num_epochs"]):
            for batch_i, seq_batch in enumerate(train_loader):
                optimizer.zero_grad()

                seq_batch = seq_batch.to(device)
                print("seq shape", seq_batch.shape) #should be batch size, seq len
                inp_batch = seq_batch
                labels_batch = torch.roll(seq_batch, -1, 1)
                labels_batch[:,-1] = 0

                #print("seq_batch", seq_batch[5])
                #print("inp batch", inp_batch[5])
                #print("labels batch", labels_batch[5])
                #exit(1)


                pred = model(inp_batch)

                flat_pred = torch.flatten(pred, start_dim=0, end_dim=1)
                flat_labels = torch.flatten(labels_batch, start_dim=0, end_dim=1)
                loss = loss_fn(flat_pred, flat_labels)
                batch_loss = loss.detach().numpy()
                batch_perp = np.exp(batch_loss)

                print("ep", epoch_i, "batch", batch_i, "loss", batch_loss, "perp", batch_perp)
                experiment.log_metric("train_loss", batch_loss)
                experiment.log_metric("train_perp", batch_perp)

                loss.backward()
                optimizer.step()





if __name__ == "__main__":
    if run_GPT:
        print("loading data for GPT model")
        vocab_size, train_loader, fine_tuning_loader, validation_loader = preprocess_GPT(gpt_hyperparams)
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
                    n=gpt_hyperparams["num_layers"]).to(device)
    else:
        model = BERT(device=device, seq_len=gpt_hyperparams["seq_len"], num_words=vocab_size,
                    d_model=gpt_hyperparams["d_model"], h=gpt_hyperparams["num_heads"],
                    n=gpt_hyperparams["num_layers"]).to(device)

    #train model
    print("training model ...")
    if run_GPT:
        train_GPT(model, train_loader, gpt_hyperparams)
    else:
        #TODO
        pass

    #fine tune model
    print("fine tuning model ...")

    #validate model
    print("validating mode ...")
