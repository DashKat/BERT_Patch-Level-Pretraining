# NOTES: Logging to wandb was removed and now it is just logged in the console

from configs import TrainConfig, DownsampledDistilBertConfig

import torch
import torch.nn as nn
import torch.optim as optim
from utils import create_model, create_dataloader
from transformers import get_linear_schedule_with_warmup

# ---- Configs (EDIT) -----

model_config = DownsampledDistilBertConfig(factors=[4,2,2,1,1,1])
train_config = TrainConfig(model_config, lr=.0002, batch_size=48)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
model, tokenizer = create_model(model_config, device)
dataset = create_dataloader(tokenizer, train_config)
optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, )
scheduler = get_linear_schedule_with_warmup(optimizer, train_config.warmup_steps, train_config.total_steps)


for step, x in enumerate(dataset):

    optimizer.zero_grad()
    x = {k : v.to(device) for k, v in x.items()}
    
    loss = model(**x).loss
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
    optimizer.step()
    scheduler.step()
    
    if step % 10 == 0:
        print("Step:" + str(step) + " Loss:" + str(loss.item()))

