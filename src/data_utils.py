import re
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def get_raw_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\r\n') for line in f]


def clean_row(row):
    text = row.lower()
    text = re.sub(r'https?://\S+|www\.\S+|t\.co/\S+|pic\.twitter\.com/\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_dataset_processed(raw_dataset):
    return [clean_row(x) for x in raw_dataset]

def get_tokinize_data(data):
    enc = tokenizer(data, add_special_tokens=True, padding=False, truncation=False)
    return enc["input_ids"]

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx], dtype=torch.long)
        }

def collate_fn(batch):
    lengths = torch.tensor([len(item['text']) for item in batch], dtype=torch.long)

    sorted_indices = torch.argsort(lengths, descending=True)
    batch = [batch[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]

    texts = [item['text'] for item in batch]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    masks = (padded_texts != 0).long()

    return {
        'texts': padded_texts,
        'masks': masks,
        'lengths': lengths
    }    