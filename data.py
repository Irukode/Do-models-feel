from torch.utils.data import Dataset
import torch
import numpy as np
import random

MAX_REVIEW_LENGTH = 192

# generates list of sentences, list of scores
def readCSV():
    reviews = []
    scores = []
    with open("IMDB_Dataset.csv", mode='r') as file:
        next(file)
        for row in file:
            sep = row.rfind(",")
            sentence = row[1:sep-1]
            score = row[sep+1:]

            sentence = sentence.replace("<br /><br />", " ")
            sentence = sentence.replace('"', "")


            if len(sentence.split()) > MAX_REVIEW_LENGTH:
                continue
            reviews.append(sentence)

            if score == "positive\n":
                score = 0
            elif score == "negative\n":
                score = 1
            else:
                print("unknown score --" + score + "--")
                print(row)
                exit(1)

            scores.append(score)
    return reviews, scores





def preprocess_GPT():
    reviews, scores = readCSV()
    tokenizer = Tokenizer()
    tokenizer.tokenize_word("PAD")

    all_ids = []
    for review in reviews:
        ids = tokenizer.tokenize_line(review)
        all_ids.append(torch.LongTensor(ids))

    seq_len = MAX_REVIEW_LENGTH #lets just use this for now

    all_ids.append(torch.zeros(seq_len)) #spacer line to ensure padding goes to seq len
    all_ids = torch.nn.utils.rnn.pad_sequence(all_ids, batch_first=True, padding_value=tokenizer.id2word("PAD"))
    all_ids=all_ids[:-1] #remove spacer line

    train_loader = training_dataset_GPT(all_ids)
    #TODO: tokenize and process scores

    #TODO: create train, fine_tune, validation loaders
    finetune_loader, validation_loader = 0, 0

    return len(tokenizer.word2id), train_loader, finetune_loader, validation_loader


class Tokenizer():
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.curr_id = 0

    def tokenize_line(self, line):
        line_ids = []
        words = line.split()
        for word in words:
            id = self.tokenize_word(word)
            line_ids.append(id)
        return line_ids

    def tokenize_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.curr_id
            self.id2word[self.curr_id] = word
            self.curr_id += 1
        return self.word2id[word]

class training_dataset_GPT(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def preprocessBERT():
    sentences, scores = readCSV()
    # TODO: tokenize sentences, scores

    #TODO: create train, fine_tune, validation loaders


# just for debugging
if __name__ == "__main__":
    sentences, scores = readCSV()
    print("first sentence", sentences[0])
    print("first score", scores[0])
    print("sum of scores", sum(scores))
    print(sentences[-30:])
    print(scores[-30:])
    print(len(sentences))
    print(len(scores))
