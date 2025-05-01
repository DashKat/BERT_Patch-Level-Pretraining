This repository contains code for our final project for CS 4650 with Dr. Weicheng Ma. The research project covers the usage of patch-level pretraining on BERT. In this repository, you can find a basic implementation of the model proposed in our final paper. For compatibility, flash-attention optimizations, wandb logging, mixed precision training, model saving, and more items were removed to create an understandable and portable implementation. Due to these factors you may see small differences in performance and relative speed increases between this version of the code and the final paper.
# File Structure
BERT_Patch-Level-Pretraining/
- 'configs.py': Defines model and training configuration classes including DownsampledDistilBertConfig which is a configuration class that stores the configuration of a DistilBertModel
- 'layers.py': Contains custom transformer layer logic. Here the 'TransformerBlock' applies the downscaling and upscaling at each layer
- 'timing.py': This benchmarks the runtime and memory usage of different transformer configurations using various compression factors and sequence lengths
- 'train.py': This runs training on the modified DistilBERT model
- 'utils.py': Utility functions for creating the model and dataloader.
  
# Basic Run Instructions
1. Open a Google Colab with an A100 (T4 should work too but is not yet tested) instance
2. !pip install datasets transformers wandb
3. Alter configuration as desired in train.py
4. python train.py
