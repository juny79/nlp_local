# placeholder for src/dataset.py
from torch.utils.data import Dataset

class SummDataset(Dataset):
    def __init__(self, tokenizer, enc, dec_in, dec_out, max_enc, max_dec):
        self.tokenizer = tokenizer
        self.enc = enc
        self.dec_in = dec_in
        self.dec_out = dec_out
        self.max_enc = max_enc
        self.max_dec = max_dec

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.enc[idx],
            max_length=self.max_enc,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        dec_in = self.tokenizer(
            self.dec_in[idx],
            max_length=self.max_dec,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        dec_out = self.tokenizer(
            self.dec_out[idx],
            max_length=self.max_dec,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "decoder_input_ids": dec_in["input_ids"].squeeze(),
            "labels": dec_out["input_ids"].squeeze(),
        }
