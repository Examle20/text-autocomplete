from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
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