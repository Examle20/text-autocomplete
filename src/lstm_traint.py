import torch
from tqdm import tqdm
import evaluate as rouge
import re

rouge = rouge.load("rouge")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

def generate_greedy(model, prefix_ids, num_tokens, eos_id=None, device="cpu"):
    model.eval()
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    logits, hidden = model(x, return_hidden=True)
    last = x[:, -1:].contiguous()
    out_ids = []
    for _ in range(num_tokens):
        step_logits, hidden = model.step(last, hidden=hidden)
        next_id = int(step_logits[:, -1, :].argmax(-1).item())
        out_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        last = last.clone()
        last[0, 0] = next_id
    return out_ids

def evaluate(model, loader, tokenizer, max_rouge_samples=400, prefix_len=10):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0.0
    pred_texts, ref_texts = [], []
    sampled = 0

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    with torch.no_grad():
        for x_batch, y_batch, attn_mask, lengths in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            logits = model(x_batch, lengths=lengths)
            V = logits.size(-1)
            loss = criterion(logits.reshape(-1, V), y_batch.reshape(-1))
            sum_loss += float(loss)

            preds = logits.argmax(dim=-1)
            mask = (y_batch != -100)
            correct += (preds[mask] == y_batch[mask]).sum().item()
            total   += mask.sum().item()

            if sampled >= max_rouge_samples:
                continue

            B = x_batch.size(0)
            for i in range(B):
                if sampled >= max_rouge_samples:
                    break

                L = int(lengths[i].item())
                if L < 2:
                    continue
                p = min(prefix_len, L - 1)
                prefix_ids = x_batch[i, :p].detach().cpu().tolist()
                y_real = y_batch[i][mask[i]].detach().cpu().tolist()
                target_ids = y_real[p:]
                if not target_ids:
                    continue

                gen_ids = generate_greedy(
                    model,
                    prefix_ids=prefix_ids,
                    num_tokens=len(target_ids),
                    eos_id=getattr(tokenizer, "eos_token_id", None),
                    device=device
                )

                pred_texts.append(norm(tokenizer.decode(gen_ids,    skip_special_tokens=True)))
                ref_texts.append( norm(tokenizer.decode(target_ids, skip_special_tokens=True)))
                sampled += 1

    if pred_texts:
        scores = rouge.compute(
            predictions=pred_texts,
            references=ref_texts,
            use_stemmer=True,
            rouge_types=["rouge1", "rouge2"]
        )
        r1, r2 = float(scores["rouge1"]), float(scores["rouge2"])
    else:
        r1 = r2 = 0.0

    val_loss = sum_loss / max(len(loader), 1)
    val_acc  = (correct / total) if total > 0 else 0.0
    return val_loss, val_acc, r1, r2

def train_model(model, train_loader, val_loader, tokenizer):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(6):
        model.train()
        train_loss = 0
        for x_batch, y_batch, attn_mask, lengths in tqdm(train_loader):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            lengths  = lengths.to(device, non_blocking=True)

            logits = model(x_batch, lengths=lengths)
            v = logits.size(-1)
            loss = criterion(logits.reshape(-1, v), y_batch.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc, r1, r2 = evaluate(model, val_loader, tokenizer)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%} | ROUGE1: {r1:.3} | ROUGE2: {r2:.3}")
