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

# need a function to compute the similarity between two embeddings
def compute_similarity(embeddings1, embeddings2):
    scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return scores.detach().numpy().tolist()

if __name__ == "__main__":
    seed = int(round(time.time()))
    label_1 = None
    label_2 = None

    def assign_label_1(label):
        global label_1
        label_1 = label

    def assign_label_2(label):
        global label_2
        label_2 = label

    figure = plt.figure(figsize=(10, 10))
    figure.suptitle("Image Similarity")

    # use matplotlib textbox to get input from user
    # - first input is label of first image
    # - second input is label of second image
    from matplotlib.widgets import TextBox
    axbox1 = figure.add_axes([0.1, 0.9, 0.8, 0.075])
    axbox2 = figure.add_axes([0.1, 0.8, 0.8, 0.075])
    text_box1 = TextBox(axbox1, "Label 1:")
    text_box2 = TextBox(axbox2, "Label 2:")
    text_box1.on_submit(assign_label_1)
    text_box2.on_submit(assign_label_2)

    plt.show(block=False)

    # randonmly select the two images based on the labels
    # provided by the user
    while label_1 is None or label_2 is None:
        plt.pause(0.05)

    # get the indices of all images with the given labels
    subset_label_1 = train_dataset.filter(lambda example: example["label"] == int(label_1))
    subset_label_2 = train_dataset.filter(lambda example: example["label"] == int(label_2))

    # randomly select two images from each subset
    image_1_idx = np.random.choice(len(subset_label_1))
    image_2_idx = np.random.choice(len(subset_label_2))

    image_1 = subset_label_1[image_1_idx]["image"]
    image_2 = subset_label_2[image_2_idx]["image"]
    
    # compute the embeddings for each image
    embeddings_1 = extract_embeddings(image_1)["embeddings"]
    embeddings_2 = extract_embeddings(image_2)["embeddings"]

    # compute the similarity between the two embeddings
    similarity = compute_similarity(embeddings_1, embeddings_2)
    
    # display the images and the similarity
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.title(f"Label: {label_1}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.title(f"Label: {label_2}")
    plt.axis("off")
    plt.suptitle(f"Similarity: {similarity}")

    plt.show(block=True)
