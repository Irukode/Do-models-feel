from comet_ml import Experiment
import torch
import torch.nn as nn
import numpy as np
from models import GPT, BERT
from data import preprocess_GPT, preprocess_BERT
from tqdm import tqdm
import argparse


run_GPT = False #TODO: toggle this depending on whether we are running BERT or GPT
run_GPT = True

gpt_hyperparams = {
    "batch_size": 4,
    "num_epochs": 4,
    "learning_rate": 0.0002,
    "num_heads": 4,
    "num_layers": 2,
    "d_model": 512,
    "seq_len": 62
 }

bert_hyperparams = {
    "batch_size": 4,
    "num_epochs": 4,
    "learning_rate": 0.004,
    "num_heads": 4,
    "num_layers": 2,
    "d_model": 256,
    "seq_len": 62
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = Experiment(project_name="final_project", log_code=True)
if (run_GPT):
    experiment.log_parameters(gpt_hyperparams)
else:
    experiment.log_parameters(bert_hyperparams)


def train_GPT(model, train_loader, hyperparams):
    return  # TODO: just for debugging finetuning loop

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    model = model.train()
    with experiment.train():
        for epoch_i in range(hyperparams["num_epochs"]):
            for batch_i, seq_batch in enumerate(train_loader):
                optimizer.zero_grad()

                seq_batch = seq_batch.to(device)
                inp_batch = seq_batch
                labels_batch = torch.roll(seq_batch, -1, 1)
                labels_batch[:,-1] = 0

                pred = model(inp_batch)

                flat_pred = torch.flatten(pred, start_dim=0, end_dim=1)
                flat_labels = torch.flatten(labels_batch, start_dim=0, end_dim=1)
                loss = loss_fn(flat_pred, flat_labels)
                batch_loss = loss.detach().cpu().numpy()
                batch_perp = np.exp(batch_loss)

                if (batch_i % 10 == 1):
                    first_preds = pred[:3].detach().cpu().numpy()
                    first_preds = first_preds.argmax(axis=2)[:5]
                    print("first preds", first_preds)
                print("ep", epoch_i, "batch", batch_i, "loss", batch_loss, "perp", batch_perp)
                experiment.log_metric("train_loss", batch_loss)
                experiment.log_metric("train_perp", batch_perp)

                loss.backward()
                optimizer.step()



def finetune_GPT(model, finetune_loader, hyperparams):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    model = model.train()
    with experiment.train():
        for epoch_i in range(hyperparams["num_epochs"]):
            for batch_i, (seq_batch, scores_batch) in enumerate(finetune_loader):
                optimizer.zero_grad()

                seq_batch = seq_batch.to(device)
                labels_batch = torch.flatten(scores_batch, start_dim=0, end_dim=1).to(device)

                pred = model(seq_batch)
                pred = pred[:,-1]

                loss = loss_fn(pred, labels_batch)
                batch_loss = loss.detach().cpu().numpy()

                labels_batch = labels_batch.cpu()
                flat_pred = torch.argmax(pred, dim=1).detach().cpu()
                total_correct = torch.sum(flat_pred==labels_batch)
                batch_acc = total_correct / float(len(labels_batch))

                batch_perp = np.exp(batch_loss)

                print("ep", epoch_i, "batch", batch_i, "loss", batch_loss, "perp", batch_perp, "acc", batch_acc)
                experiment.log_metric("finetune_loss", batch_loss)
                experiment.log_metric("finetune_perp", batch_perp)
                experiment.log_metric("finetune_acc", batch_acc)

                loss.backward()
                optimizer.step()

                if batch_i > 250: #TODO: debugging, remove this
                    return


def validate_GPT(model, validate_loader, hyperparams):
    loss_fn = nn.CrossEntropyLoss()
    total_token_count = 0
    total_correct_token_count = 0

    model = model.eval()
    with experiment.test():
        for batch_i, (seq_batch, scores_batch) in enumerate(validate_loader):
            seq_batch = seq_batch.to(device)
            labels_batch = torch.flatten(scores_batch, start_dim=0, end_dim=1).to(device)

            pred = model(seq_batch)
            pred = pred[:, -1]

            loss = loss_fn(pred, labels_batch)
            batch_loss = loss.detach().cpu().numpy()

            labels_batch = labels_batch.cpu()
            flat_pred = torch.argmax(pred, dim=1).detach().cpu()
            batch_correct = torch.sum(flat_pred == labels_batch)
            batch_acc = batch_correct / float(len(labels_batch))

            batch_perp = np.exp(batch_loss)

            print("batch", batch_i, "loss", batch_loss, "perp", batch_perp, "acc", batch_acc)
            experiment.log_metric("test_loss", batch_loss)
            experiment.log_metric("test_perp", batch_perp)
            experiment.log_metric("test_acc", batch_acc)

            total_token_count += len(seq_batch)
            total_correct_token_count += batch_correct

        experiment.log_metric("test_final_acc", total_correct_token_count / float(total_token_count))


def train_BERT(model, train_loader, hyperparams):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    model = model.train()
    with experiment.train():
        for epoch_i in range(hyperparams["num_epochs"]):
            total_loss = 0
            word_count = 0
            for batch_i in tqdm(train_loader):
                optimizer.zero_grad()

                seq_batch = batch_i["sequences"].to(device)
                masked_indices = batch_i["masked_indices"].to(device)
                labels = batch_i["labels"].to(device)
                # print("seq shape", seq_batch.shape)  # should be [batch size, seq len]

                # grab the labels we need for loss calculation
                masked_y_labels = torch.LongTensor(np.zeros(masked_indices.shape)).to(device)
                for row in range(masked_indices.shape[0]):
                    masked_y_labels[row] = labels[row][masked_indices[row]]

                current_count = masked_indices.shape[0] * masked_indices.shape[1]
                word_count += current_count

                pred = model(seq_batch)

                y_pred_masked = torch.FloatTensor(
                    np.zeros((masked_indices.shape[0], masked_indices.shape[1], pred.shape[2]))).to(device)
                for row in range(masked_indices.shape[0]):
                    y_pred_masked[row] = pred[row][masked_indices[row]]

                flat_pred = y_pred_masked.permute(0,2,1)

                loss = loss_fn(flat_pred, masked_y_labels)
                total_loss += loss * current_count

                batch_loss = loss.detach().cpu().numpy()
                batch_perp = np.exp(batch_loss)

                # print("ep", epoch_i, "loss", batch_loss, "perp", batch_perp)
                experiment.log_metric("train_loss", batch_loss)
                experiment.log_metric("train_perp", batch_perp)

                loss.backward()
                optimizer.step()

            average_loss = total_loss / word_count
            perplexity = torch.exp(average_loss)
            perplexity = perplexity.detach().cpu().clone().numpy()
            print("ep", epoch_i, "loss", total_loss, "perp", perplexity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    if run_GPT:
        print("loading data for GPT model")
        vocab_size, train_loader, fine_tuning_loader, validation_loader = preprocess_GPT(gpt_hyperparams)
    else:
        print("loading data for BERT model")
        vocab_size, train_loader, fine_tuning_loader, validation_loader = preprocess_BERT(bert_hyperparams)

    #create model
    if run_GPT:
        model = GPT(device=device, seq_len=gpt_hyperparams["seq_len"], num_words=vocab_size,
                    d_model=gpt_hyperparams["d_model"], h= gpt_hyperparams["num_heads"],
                    n=gpt_hyperparams["num_layers"]).to(device)

        if args.load:
            model.load_state_dict(torch.load('./gpt_model.pt'))
    else:
        model = BERT(device=device, seq_len=bert_hyperparams["seq_len"], num_words=vocab_size,
                    d_model=bert_hyperparams["d_model"], h=bert_hyperparams["num_heads"],
                    n=bert_hyperparams["num_layers"]).to(device)
        if args.load:
            model.load_state_dict(torch.load('./bert_model.pt'))

    #train model
    print("training model ...")
    if run_GPT:
        train_GPT(model, train_loader, gpt_hyperparams)
        if args.save:
            torch.save(model.state_dict(), './gpt_model.pt')
    else:
        train_BERT(model, train_loader, bert_hyperparams)
        if args.save:
            torch.save(model.state_dict(), './bert_model.pt')

    #fine tune model
    print("fine tuning model ...")
    if run_GPT:
        finetune_GPT(model, fine_tuning_loader, gpt_hyperparams)
    else:
        pass

    #validate model
    print("validating mode ...")
    if run_GPT:
        validate_GPT(model, validation_loader, gpt_hyperparams)
    else:
        pass

