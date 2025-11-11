import torch
from tqdm.auto import tqdm
import evaluate as eval_r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


rouge_metric = eval_r.load("rouge")

def evaluate_rouge(model, loader, tokenizer, pad_id=0, device=None, max_batches=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for bi, (X, Y, lengths) in enumerate(loader):
            if max_batches is not None and bi >= max_batches: break
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            for b in range(X.size(0)):
                seq = torch.cat([X[b][X[b]!=pad_id], Y[b][Y[b]!=pad_id][-1:]])
                if seq.numel() < 4: 
                    continue
                k = max(1, int(0.75 * seq.numel()))
                prefix_ids = seq[:k].tolist()
                target_ids = seq[k:].tolist()

                prompt = tokenizer.decode(prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                reference = tokenizer.decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                generated = model.generate(tokenizer, prompt, max_new_tokens=len(target_ids), temperature=0.0, top_p=None)
                preds.append(generated)
                refs.append(reference)

    if not preds:
        return 0.0, 0.0, 0
    scores = rouge_metric.compute(
        predictions=preds, references=refs,
        rouge_types=["rouge1","rouge2"], use_stemmer=True
    )
    return float(scores["rouge1"]), float(scores["rouge2"])

def evaluate(model, loader, pad_id=0):
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch, lengths in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                logits, _ = model(x_batch, lengths)
                V = logits.size(-1)
                loss = criterion(logits.reshape(-1, V), y_batch.reshape(-1))

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = (y_batch != pad_id)
            correct += (preds[mask] == y_batch[mask]).sum().item()
            total += mask.sum().item()

    val_loss = total_loss / max(1, total)
    val_acc = (correct / total) if total > 0 else 0.0
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, tokenizer, epochs=6, lr=1e-3, pad_id=0):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss, total_tokens = 0.0, 0
        scaler = torch.amp.GradScaler(enabled=(device=="cuda"))
        for x_batch, y_batch, lengths in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                logits, _ = model(x_batch, lengths)
                v = logits.size(-1)
                loss = criterion(logits.reshape(-1, v), y_batch.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            n_tok = (y_batch != pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        train_loss = total_loss / max(1, total_tokens)  
        val_loss, val_acc = evaluate(model, val_loader)
        r1, r2 = evaluate_rouge(model, val_loader, tokenizer, pad_id=pad_id, device=device, max_batches=20)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%} | ROUGE1: {r1:.4f} | ROUGE2: {r2:.4f}")
