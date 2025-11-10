import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd, csv

def clean_row(row):
    text = (row or "").lower().strip()
    text = re.sub(r'(https?://\S+|www\.\S+|t\.co/\S+|pic\.twitter\.com/\S+)', ' <url> ', text, flags=re.IGNORECASE)
    text = re.sub(r'@\w+', ' <user> ', text)
    text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)
    text = re.sub(r'#(\w+)', r' <hashtag> \1 ', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'\s+([\.\,\!\?\:\;])', r'\1', text)
    text = re.sub(r'[^a-z0-9\.\,\!\?\:\;\'\"\-\s]', ' ', text)
    text = re.sub(r'([!?])\1{2,}', r'\1\1', text)
    text = text.replace('’', "'")
    text = re.sub(r'&quot;?', '"', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s*', ', ', text)
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
        dtype=str,
        keep_default_na=False
    )
    return list(map(clean_row, df['text']))

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=128, stride=None):
        self.samples = []
        self.pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        bos_id = tokenizer.cls_token_id
        eos_id = tokenizer.sep_token_id
        stride = seq_len if stride is None else max(1, stride)
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False)
            token_ids = [bos_id] + token_ids
            token_ids = token_ids + [eos_id]
            for i in range(0, len(token_ids) - 1, stride):
                chunk = token_ids[i:i + seq_len + 1]
                if len(chunk) < 2:
                    continue
                self.samples.append((chunk[:-1], chunk[1:]))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    X = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    Y = pad_sequence(ys, batch_first=True, padding_value=pad_id)
    lengths = torch.as_tensor([len(x) for x in xs], dtype=torch.long)
    return X, Y, lengths