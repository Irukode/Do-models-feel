from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm

MAX_WORDS = 126 #seq len - 2
MASK_TOKEN = "<MASK>"

# generates sentences and scores for train, fine tune, validation
def readCSV():
    reviews = []
    scores = []
    with open("IMDB_Dataset.csv", mode='r', encoding='utf-8') as file:
        next(file)
        for row in tqdm(file):
            sep = row.rfind(",")
            sentence = row[1:sep-1]
            score = row[sep+1:]

            sentence = sentence.replace("<br /><br />", " ")
            sentence = sentence.replace('"', "")


            reviews.append(sentence)

            assert score == "positive\n" or score == "negative\n"
            scores.append(score)

    assert len(reviews) == len(scores)
    reviews = np.array(reviews)
    scores = np.array(scores)

    indices = np.arange(len(reviews))
    np.random.shuffle(indices)
    length = len(reviews)
    train_indices = indices[:int(length * 0.7)]
    finetune_indices = indices[int(length*0.7):int(length * 0.9)]
    validate_indices = indices[int(length*0.9):]

    train_reviews = reviews[train_indices]
    finetune_reviews = reviews[finetune_indices]
    finetune_scores = scores[finetune_indices]
    validate_reviews = reviews[validate_indices]
    validate_scores = scores[validate_indices]

    return train_reviews, finetune_reviews, finetune_scores, validate_reviews, validate_scores

class Tokenizer():
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.curr_id = 0
        self.tokenize_word("PAD")

    def tokenize_reviews_list(self, reviews, seq_len):
        all_ids = []
        for review in tqdm(reviews):
            ids = self.tokenize_line(review)
            all_ids.append(torch.LongTensor(ids))

        all_ids.append(torch.zeros(seq_len))  # spacer line to ensure padding goes to seq len
        all_ids = torch.nn.utils.rnn.pad_sequence(all_ids, batch_first=True, padding_value=self.word2id["PAD"])
        all_ids = all_ids[:-1]  # remove spacer line
        assert len(all_ids[0]) == seq_len
        return all_ids

    def tokenize_line(self, line):
        line_ids = []
        words = line.split()[:MAX_WORDS] #truncate it to fit in sequence
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


def preprocess_GPT(hyperparams):
    train_reviews, finetune_reviews, finetune_scores, validate_reviews, validate_scores = readCSV()
    tokenizer = Tokenizer()

    train_reviews = tokenizer.tokenize_reviews_list(train_reviews, hyperparams["seq_len"])
    finetune_reviews = tokenizer.tokenize_reviews_list(finetune_reviews, hyperparams["seq_len"])
    finetune_scores = tokenizer.tokenize_reviews_list(finetune_scores, 1)
    validate_reviews = tokenizer.tokenize_reviews_list(validate_reviews, hyperparams["seq_len"])
    validate_scores = tokenizer.tokenize_reviews_list(validate_scores, 1)

    train_set = training_dataset_GPT(train_reviews)
    finetune_set = finetune_dataset_GPT(finetune_reviews, finetune_scores)
    validation_set = finetune_dataset_GPT(validate_reviews, validate_scores)

    #TODO: create train, fine_tune, validation loaders
    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=False) #TODO: set to true
    finetune_loader = DataLoader(finetune_set, batch_size=hyperparams["batch_size"], shuffle=False) #TODO: set to true
    validation_loader = DataLoader(validation_set, batch_size=hyperparams["batch_size"], shuffle=False)

    return len(tokenizer.word2id), train_loader, finetune_loader, validation_loader

class finetune_dataset_GPT(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores
        assert len(sequences) == len(scores)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]

class training_dataset_GPT(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        print("train dataset shape", sequences.shape)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def preprocess_BERT(hyperparams):
    train_reviews, finetune_reviews, finetune_scores, validate_reviews, validate_scores = readCSV()
    tokenizer = Tokenizer()
    # TODO: tokenize sentences, scores
    train_reviews = tokenizer.tokenize_reviews_list(train_reviews, hyperparams["seq_len"])
    finetune_reviews = tokenizer.tokenize_reviews_list(finetune_reviews, hyperparams["seq_len"])
    finetune_scores = tokenizer.tokenize_reviews_list(finetune_scores, 1)
    validate_reviews = tokenizer.tokenize_reviews_list(validate_reviews, hyperparams["seq_len"])
    validate_scores = tokenizer.tokenize_reviews_list(validate_scores, 1)

    #TODO: create train, fine_tune, validation loaders
    word2id = tokenizer.word2id
    word2id[MASK_TOKEN] = len(word2id)

    train_set = training_dataset_BERT(train_reviews, word2id)
    finetune_set = finetune_dataset_BERT(finetune_reviews, word2id, finetune_scores)
    validate_set = finetune_dataset_BERT(validate_reviews, word2id, validate_scores)

    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=False) # TODO: set to true
    finetune_loader = DataLoader(finetune_set, batch_size=hyperparams["batch_size"], shuffle=False) # TODO: set to true
    validation_loader = DataLoader(validate_set, batch_size=hyperparams["batch_size"], shuffle=False) # TODO: set to true

    return len(word2id), train_loader, finetune_loader, validation_loader

class training_dataset_BERT(Dataset):
    def __init__(self, sequences, word2id):
        self.labels = sequences.numpy()
        self.sequences = sequences.tolist()
        self.word2id = word2id
        self.masked_indices = []

        seq_length = len(self.sequences[0])

        total_pick = round(seq_length * 0.15)

        mask_id = self.word2id[MASK_TOKEN]

        # for every sequence, mask out 15% of the tokens
        for rows in tqdm(self.sequences):
            rand_indices = torch.randperm(seq_length)
            self.masked_indices.append(rand_indices[:total_pick].tolist())
            for i in range(total_pick):
                # mask logic
                rand_num = np.random.rand(1)
                if(rand_num < 0.8):
                    rows[rand_indices[i]] = mask_id
                elif(rand_num < 0.9):
                    rows[rand_indices[i]] = np.random.randint(len(self.word2id))

        self.sequences = torch.LongTensor(self.sequences)
        self.masked_indices = torch.LongTensor(self.masked_indices)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequences": self.sequences[idx],
            "masked_indices": self.masked_indices[idx],
            "labels": self.labels[idx]
        }
        return item

class finetune_dataset_BERT(Dataset):
    def __init__(self, sequences, word2id, scores):
        self.labels = sequences.numpy()
        self.sequences = sequences.tolist()
        self.word2id = word2id
        self.scores = scores
        assert len(sequences) == len(scores)
        self.masked_indices = []

        seq_length = len(self.sequences[0])

        total_pick = round(seq_length * 0.15)

        mask_id = self.word2id[MASK_TOKEN]

        # for every sequence, mask out 15% of the tokens
        for rows in tqdm(self.sequences):
            rand_indices = torch.randperm(seq_length)
            self.masked_indices.append(rand_indices[:total_pick].tolist())
            for i in range(total_pick):
                # mask logic
                rand_num = np.random.rand(1)
                if (rand_num < 0.8):
                    rows[rand_indices[i]] = mask_id
                elif (rand_num < 0.9):
                    rows[rand_indices[i]] = np.random.randint(len(self.word2id))

        self.sequences = torch.LongTensor(self.sequences)
        self.masked_indices = torch.LongTensor(self.masked_indices)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequences": self.sequences[idx],
            "masked_indices": self.masked_indices[idx],
            "labels": self.labels[idx],
            "scores": self.scores[idx]
        }
        return item

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
