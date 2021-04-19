from torch.utils.data import Dataset
import torch
import numpy as np
import random


# generates list of sentences, list of scores
def readCSV():
    sentences = []
    scores = []
    with open("IMDB_Dataset.csv", mode='r') as file:
        next(file)
        for row in file:
            sep = row.rfind(",")
            sentence = row[1:sep-1]
            score = row[sep+1:]

            sentence = sentence.replace("<br /><br />", " ")
            sentence = sentence.replace('"', "")


            if len(sentence.split()) > 192:
                continue
            sentences.append(sentence)

            if score == "positive\n":
                score = 0
            elif score == "negative\n":
                score = 1
            else:
                print("unknown score --" + score + "--")
                print(row)
                exit(1)

            scores.append(score)
    return sentences, scores

def preprocessGPT():
    sentences, scores = readCSV()
    #TODO: tokenize sentences, scores

    #TODO: get vocab size
    vocab_size = 0

    #TODO: create train, fine_tune, validation loaders
    train_loader, finetune_loader, validation_loader = 0, 0, 0

    return vocab_size, train_loader, finetune_loader, validation_loader



# def preprocessBERT():
#     sentences, scores = readCSV()
#     # TODO: tokenize sentences, scores
#
#     #TODO: create train, fine_tune, validation loaders


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
