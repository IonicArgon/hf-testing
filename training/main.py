import torch
import numpy as np

from transformers import ViTModel, ViTImageProcessor
from datasets import load_dataset

# we're gonna try to train a base ViT model on food101

model_checkpoint = 'google/vit-base-patch16-224'
model = ViTModel.from_pretrained(model_checkpoint)
processor = ViTImageProcessor.from_pretrained(model_checkpoint)

dataset = load_dataset('food101')
train_dataset = dataset['train']
validation_dataset = dataset['validation']


