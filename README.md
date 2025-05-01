# Overview
This repository contains code for our final project for CS 4650 with Dr. Weicheng Ma. The research project covers the usage of patch-level pretraining on BERT. In this repository, you can find a basic implementation of the model proposed in our final paper. Note that this is a slightly different implementation compared to the code used to generate our results. For compatibility, flash-attention optimizations, wandb logging, mixed precision training, model saving, and more items were removed to create an understandable and portable implementation. Due to these factors you may see small differences in performance and relative speed differences between this version of the code and the final paper. 

This codebase implements a modified version of DistilBERT that introduces layer-wise compression to reduce training costs. Instead of processing full token sequence at every layer, each transformer block will temporarily downsample the input sequence, perform attention and feedforward operations, and upsample back to the original length. By doing this, we aim to improve the speed of training and decrease memory usage while maintaining relatively similar performance to the standard DistilBERT model.

# Dataset
This model was trained on the FineWeb-Edu dataset, which contains 1.3 trillion tokens of educational data such as research papers, allowing for higher performance on reasoning and knowledge-based tasks. The dataset used can be found [HERE](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

Full path: `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu`

# File Structure
BERT_Patch-Level-Pretraining/
- `configs.py`: Defines model and training configuration classes including DownsampledDistilBertConfig which is a configuration class that stores the configuration of a DistilBertModel
- `layers.py`: Contains custom transformer layer logic. Here the 'TransformerBlock' applies the downscaling and upscaling at each layer
- `timing.py`: This benchmarks the runtime and memory usage of different transformer configurations using various compression factors and sequence lengths
- `train.py`: This runs training on the modified DistilBERT model
- `utils.py`: Utility functions for creating the model and dataloader.
  
# Basic Run Instructions
1. Open a Google Colab with an A100 (T4 will run out of memory, may work if batch size is lowered) instance. 
2. `!pip install datasets transformers wandb` (If not using a colab, other packages may need to be installed).
3. Alter configuration classes as desired in `train.py` as desired.
4. Run `python train.py` to see the loss decrease with the pretraining process.
5. Alter configuration classes as desired in `timing.py` as desired.
6. Run `python timing.py` to see timing figures.

# Group Members

Pranav Devarinti, Dasarath (Dash) Katragadda, Will Tjokroamidjojo, Nikhil Sathisha
