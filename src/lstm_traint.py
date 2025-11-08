
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            v = logits.size(-1)

            loss = criterion(logits.reshape(-1, v), y_batch.reshape(-1))
            sum_loss += loss.item()

            preds = logits.argmax(dim=-1)

            mask = (y_batch != -100)
            correct += (preds[mask] == y_batch[mask]).sum().item()
            total += mask.sum().item()
        return sum_loss / max(len(loader), 1), (correct / total if total > 0 else 0.0)

def train_model(model, train_loader, val_loader):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(2):
        model.train()
        train_loss = 0
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            v = logits.size(-1)
            loss = criterion(logits.reshape(-1, v), y_batch.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")