from transformers import AutoTokenizer, DistilBertForMaskedLM, DataCollatorForLanguageModeling

from datasets import load_dataset

from layers import TransformerBlock

from torch.utils.data import IterableDataset, DataLoader

import math



def create_model(model_config, device):
    model = DistilBertForMaskedLM(model_config)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    for layer_idx, factor in enumerate(model_config.factors):
        block = TransformerBlock(model_config, layer_idx)
        model.distilbert.transformer.layer[layer_idx] = block
    
    model.to(device)
    return model, tokenizer

def create_dataloader(tokenizer, train_config):
    
    def tokenize(row):
        return tokenizer(row["text"], truncation=True, padding="max_length", max_length=train_config.max_seq)

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", split="train", streaming = True)
    remove_cols = [c for c in dataset.column_names if c not in ["input_ids", "attention_mask"]]
    tokenized = dataset.map(tokenize, batched=True, remove_columns=remove_cols)

    pad_mult = math.gcd(*train_config.model_config.factors)

    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=train_config.mlm_probability, pad_to_multiple_of=pad_mult)
    return DataLoader(tokenized, batch_size=train_config.batch_size, collate_fn=collate_fn)