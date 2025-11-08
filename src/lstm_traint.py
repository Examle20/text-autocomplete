import torch
from tqdm import tqdm
import evaluate

rouge = evaluate.load("rouge")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

def _generate_greedy(model, prefix_ids, num_tokens, eos_id=None):
    model.eval()
    x = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = []
    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(x)[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            out.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    return out

def evaluate(model, loader, tokenizer):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0

    pred_texts, ref_texts = [], []

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

            B = x_batch.size(0)
            for i in range(B):
                labels_i = y_batch[i]
                mask_i = mask[i]
                y_ids = labels_i[mask_i].tolist()
                if not y_ids:
                    continue
                pos = (labels_i != -100).nonzero(as_tuple=False)
                if pos.numel() == 0:
                    continue
                start = int(pos[0].item())

                prefix_ids = x_batch[i, :start].tolist()
                if len(prefix_ids) < 1:
                    continue

                target_ids = y_ids

                gen_ids = _generate_greedy(
                    model,
                    prefix_ids,
                    num_tokens=len(target_ids),
                    eos_id=getattr(tokenizer, "eos_token_id", None)
                )
                pred_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                ref_texts.append(tokenizer.decode(target_ids, skip_special_tokens=True).strip())

        if len(pred_texts) > 0:
            scores = rouge.compute(
                predictions=pred_texts,
                references=ref_texts,
                use_stemmer=True,
                rouge_types=["rouge1", "rouge2"]
            )
            r1, r2 = float(scores["rouge1"]), float(scores["rouge2"])
        else:
            r1 = r2 = 0.0

        return sum_loss / max(len(loader), 1), (correct / total if total > 0 else 0.0), r1, r2

def train_model(model, train_loader, val_loader, tokenizer):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(10):
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
        val_loss, val_acc, r1, r2 = evaluate(model, val_loader, tokenizer)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%} | ROGUE1: {r1:.2%} | ROGUE2: {r2:.2%}")
