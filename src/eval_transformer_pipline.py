from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate as eval_r
import torch

rouge_metric = eval_r.load("rouge")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name)


generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)


def transformer_generate(prompts, max_new_tokens=20, min_new_tokens=5):
    outs = generator(
        prompts,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1
    )   
    for i, (p, o) in enumerate(zip(prompts, outs), 1):
        text = o[0]["generated_text"]
        print(f"\n[{i}] PROMPT:\n{p}\n---\nOUTPUT:\n{text}\n" + "-"*60)


def evaluate_rouge_distilgpt2(loader, lstm_tok, device="cuda", pad_id=0, deterministic=False, max_batches=None):
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
                prefix_text = lstm_tok.decode(prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                ref_text  = lstm_tok.decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if not ref_text.strip():
                    continue
                enc = tokenizer([prefix_text], return_tensors="pt", padding=False, truncation=False)
                enc = {k: v.to(device) for k, v in enc.items()}
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=max(5, len(target_ids)),
                    min_new_tokens=5,
                    do_sample=(not deterministic),
                    temperature=0.8 if not deterministic else None,
                    top_p=0.95 if not deterministic else None,
                    early_stopping=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )[0]
                new_ids = gen_ids[enc["input_ids"].size(1):]
                pred_text = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                preds.append(pred_text)
                refs.append(ref_text)
    if not preds:
        return 0.0, 0.0, 0

    scores = rouge_metric.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rouge1","rouge2"],
        use_stemmer=True
    )
    return float(scores["rouge1"]), float(scores["rouge2"])
