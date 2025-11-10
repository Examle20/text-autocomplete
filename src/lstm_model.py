import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

torch.backends.cudnn.benchmark = True
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

    def forward(self, x, lengths=None, hidden=None):
        emb = self.embedding(x)
        h0 = c0 = None
        if hidden is not None:
            h0, c0 = hidden
        if lengths is not None:
            packed = rnn_utils.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, (h, c) = self.rnn(packed, (h0, c0) if h0 is not None else None)
            out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        else:
            out, (h, c) = self.rnn(emb, (h0, c0) if h0 is not None else None)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, (h, c)
    
    def generate(self, tokenizer, prompt, max_new_tokens=40, temperature=0.9, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            ids = tokenizer.encode((prompt or ""), add_special_tokens=False)
            bos_id = tokenizer.cls_token_id
            eos_id = tokenizer.sep_token_id

            tokens = [bos_id] + ids if bos_id is not None else ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

            _, hidden = self.forward(tokens, hidden=None)

            for _ in range(max_new_tokens):
                last_token = tokens[:, -1:]
                logits, hidden = self.forward(last_token, hidden=hidden)
                logits = logits[:, -1, :].squeeze(0) / max(temperature, 1e-8)

                if top_p and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumsum = torch.cumsum(probs, dim=-1)
                    cutoff = cumsum > top_p
                    cutoff[1:] = cutoff[:-1].clone(); cutoff[0] = False
                    sorted_logits[cutoff] = float('-inf')
                    logits = torch.full_like(logits, float('-inf')).scatter(0, sorted_idx, sorted_logits)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).view(1, 1)
                tokens = torch.cat([tokens, next_token.to(device)], dim=1)

                if eos_id is not None and next_token.item() == eos_id:
                    break

            out = tokens[0].tolist()
            if bos_id is not None and out and out[0] == bos_id:
                out = out[1:]
            return tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)