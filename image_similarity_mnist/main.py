# attempting to use a model to compute similarity between
# two random images pulled from the MNIST dataset

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time

from transformers import ViTModel, ViTImageProcessor
from datasets import load_dataset

#? we'll break stuff out into other files as we go along
#? for now, we'll just do everything in main.py

# load our model and feature extractor
model_checkpoint = "farleyknight-org-username/vit-base-mnist"
model = ViTModel.from_pretrained(model_checkpoint)
processor = ViTImageProcessor.from_pretrained(model_checkpoint)

# load our dataset
dataset = load_dataset("mnist")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
print("Dataset loaded.")

# we shouldn't need a transformation chain, since the images are already
# the correct size and shape

# we'll need to extract embeddings from the images
def extract_embeddings(image):
    global model
    global processor

    image_np = np.array(image)
    image_rgb = np.repeat(image_np[None, :, :], 3, axis=0)

    inputs = processor(images=image_rgb, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0].cpu()
    return {
        "embeddings": embeddings,
    }

# let's just test this function by grabbing a random image from the dataset
# and extracting its embeddings, printing them out
# we'll also plot the image

seed = int(round(time.time()))
random_image = train_dataset.shuffle(seed=seed).select(range(1))
image = random_image["image"][0]
label = random_image["label"][0]
print("Image loaded.")

# plot the image
plt.imshow(image, cmap="gray")
plt.title(f"Label: {label}")
plt.show(block=False) # we'll add a block to the terminal later

# extract the embeddings
embeddings = extract_embeddings(image)["embeddings"]
print("Embeddings extracted.")

# print the embeddings
print("Embeddings:")
print(embeddings)
input("Press any key to continue...")
