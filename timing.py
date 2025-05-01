import time
import torch
import torch.nn as nn
import pandas as pd
from layers import FFN, TransformerBlock
from configs import TimingConfig, DownsampledDistilBertConfig

def benchmarkLayer(layer, embeds, mask=None, numWarmups=10, numRepeats=50):
    curDevice = embeds.device
    torch.cuda.reset_peak_memory_stats(curDevice)

    numIters = numWarmups + numRepeats
    runtimes = []
    for i in range(numIters):
        torch.cuda.synchronize()
        startTime = time.time()

        # Does this work or do we need to check for downscaleFactor?
        _ = layer(embeds, attn_mask=mask)

    # Remove warmup from final timings
    avgTime = sum(runtimes[numWarmups:]) / len(runtimes[numWarmups:])
    memoryUsed = torch.cuda.max_memory_allocated(curDevice)

    return avgTime, memoryUsed

config = TimingConfig(contextLengths=[256, 512, 1024, 2048],
                      batchSizes=[16, 32, 64],
                      factors=[1, 2, 4, 8],
                      dModel=768,
                      nhead=12,
                      feedforwardDim=3072,
                      dropout=0.0,
                      attnDropout=0.0,
                      activation='gelu') #ignored, but i'll pass to config anyway
                     
device = torch.device('cuda')

output = []
for contextLength in config.contextLengths:
    for batchSize in config.batchSizes:
        embedding = torch.randn(batchSize, contextLength, config.dModel, device=device)
        mask = None
        
        for factor in config.factors: #Factor of 1 can be used as the baseline. It should be equivalent to the original DistillBERT layer w/ no computations apart from fast if statement evals. 
            # How do i generate config for this
            layer_config = DownsampledDistilBertConfig(dim=config.dModel,
                                                      n_heads=config.nhead,
                                                      n_layers=1,
                                                      hidden_dim=config.feedforwardDim,
                                                      dropout=config.dropout,
                                                      attention_dropout=config.attnDropout,
                                                      activation=config.activation,
                                                      max_seq=contextLength,
                                                      factors=[factor])
            
            layer = TransformerBlock(layer_config, 0)
            compiledLayer = torch.compile(layer)

            customTime, customMem = benchmarkLayer(compiledLayer, embedding, mask)
            output.append({
                'layer': 'custom',
                'factor': factor,
                'bs': batchSize,
                'L': contextLength,
                'time': customTime,
                'mem': customMem
            })

df = pd.DataFrame(output)

# Generate output tables

# time
timeTable = df.pivot_table(index=['L', 'bs'], columns='factor', values='time')

# memory
memTable = df.pivot_table(index=['L', 'bs'], columns='factor', values='mem')

# Calculate norms manually

print(f"Runtime (seconds):\n {timeTable}")
print(f"\n\n Memory Usage (bytes):\n{memTable}")
