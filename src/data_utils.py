import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd, csv

def clean_row(row):
    text = row.lower()
    text = re.sub(r'https?://\S+|www\.\S+|t\.co/\S+|pic\.twitter\.com/\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_clean_text(path):
    df = pd.read_csv(
        path,
        header=None,
        names=["text"],
        sep="\x01",
        engine="python",
        quoting=csv.QUOTE_NONE,
        escapechar="\\", 
        skip_blank_lines=False,
        dtype=str
    )
    return list(map(clean_row, df['text']))

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.samples = []
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)
            if len(token_ids) < 2:
                continue
            x = token_ids[:-1]
            y = token_ids[1:]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    xs = [torch.tensor(x, dtype=torch.long) for x in xs]
    ys = [torch.tensor(y, dtype=torch.long) for y in ys]
    X = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    Y = pad_sequence(ys, batch_first=True, padding_value=-100)
    return X, Y