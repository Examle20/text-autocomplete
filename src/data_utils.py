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
            for i in range(1, len(token_ids) - 1):
                input = token_ids[: i + 1]
                target = token_ids[i + 1:]
                self.samples.append((input, target))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    lengths = torch.tensor([len(item['text']) for item in batch], dtype=torch.long)

    sorted_indices = torch.argsort(lengths, descending=True)
    batch = [batch[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]

    texts = [item['text'] for item in batch]
    print(texts)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    masks = (padded_texts != 0).long()

    return {
        'texts': padded_texts,
        'masks': masks,
        'lengths': lengths
    }    