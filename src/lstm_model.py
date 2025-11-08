import torch.nn as nn

class LSTMAutoCopleteText(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, emb_dim = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=True)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        linear_out = self.fc(out)
        return linear_out