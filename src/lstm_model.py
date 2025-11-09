import torch
import torch.nn as nn

class LSTMAutoCopleteText(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=128,
        emb_dim=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=True)
        self.fc.weight = self.embedding.weight

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden
    
    def generate(self, tokenizer, prompt, max_length=20, device='cpu'):
        self.eval()
        self.to(device)
        with torch.no_grad():
            tokens = tokenizer.encode(prompt.lower(), return_tensors='pt').to(device)
            generated = tokens.clone()
           
            logits, hidden = self.forward(tokens, hidden)

            for _ in range(max_length - tokens.size(1)):
                last_token = generated[:, -1:].to(device)
                logits, hidden = self.forward(last_token, hidden)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is not None and next_token.item() == eos_id:
                    break

            return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)