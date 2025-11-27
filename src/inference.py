# placeholder for src/inference.py
# inference.py
import torch
from torch.utils.data import DataLoader

def generate_summary(model, tokenizer, dataset, cfg):
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    outputs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(cfg.device)
            mask = batch["attention_mask"].to(cfg.device)

            gen = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=100,
                num_beams=cfg.num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.7,
                length_penalty=1.0,
                early_stopping=True
            )

            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            outputs.extend(decoded)

    return outputs
