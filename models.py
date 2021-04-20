from torch import nn
import torch


#A Transformer Language Model
class GPT(nn.Module):
    def __init__(self, device, seq_len, num_words, d_model=512, h=8, n=6):
        super().__init__()
        self.device = device
        self.tok_emb = nn.Embedding(num_embeddings=num_words, embedding_dim=d_model)
        self.pos_emb = nn.Parameter(torch.randn([seq_len, d_model]),requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead=h, dim_feedforward=seq_len, dropout=0.1, activation='relu')
        # self.mask = (-1) * float("inf") * torch.ones([seq_len,
        #                                               seq_len])
        # self.mask = self.mask.triu(diagonal=1).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n)

        self.linear = nn.Linear(d_model, num_words)


    def forward(self, x):
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        out = self.tok_emb(x) + self.pos_emb

        out = out.transpose(1,0) #Transformer requires batch second!

        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device)

        out = self.transformer(out, mask=mask)

        out = out.transpose(0,1) #convert back to batch first!

        out = self.linear(out)
        return out


#A Transformer Language Representation Model
class BERT(nn.Module):
    def __init__(self, device, seq_len, num_words, d_model=512, h=8, n=6):
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=num_words, embedding_dim=d_model)
        self.pos_emb = nn.Parameter(torch.randn([seq_len, d_model]),requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead=h, dim_feedforward=seq_len, dropout=0.1, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n)
        self.linear = nn.Linear(d_model, num_words)

    def forward(self, x):
        out = self.tok_emb(x) + self.pos_emb
        out = out.transpose(1,0) #Transpose to put the batch second?
        out = self.transformer(out)
        out = out.transpose(1,0)
        out = self.linear(out)
        return out


