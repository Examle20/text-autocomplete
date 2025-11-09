import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMAutoCopleteText(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=128,
        emb_dim=128,
        num_layers=1,
        pad_id=0,
        dropout=0.0
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.0 if num_layers == 1 else dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=True)
        self.fc.weight = self.embedding.weight

    def forward(self, x, lengths=None, hidden=None, return_hidden=False):
        emb = self.embedding(x)
        if lengths is not None:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, hidden = self.rnn(packed, hidden)
            out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))
        else:
            out, hidden = self.rnn(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)

        if return_hidden:
            return logits, hidden
        return logits
    
    @torch.no_grad()
    def step(self, last_token_ids, hidden=None):
        emb = self.embedding(last_token_ids)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden