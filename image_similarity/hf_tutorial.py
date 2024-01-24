from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision as tv

# load model and image processor
model_id = "nateraw/vit-base-beans" # this is literally just
                                    # a ViT model trained on the beans dataset
extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
print("Model loaded.")

# load dataset
dataset = load_dataset("beans")
print("Dataset loaded.")

# set up dictionaries for later
labels = dataset["train"].features["labels"].names
labels_to_idx, idx_to_labels = {}, {}

for i, label in enumerate(labels):
    labels_to_idx[label] = i
    idx_to_labels[i] = label

# a few parameters
num_samples = 100
seed = int(round(time.time()))

# get a random subset of the dataset
candidate_image_subset = dataset["train"].shuffle(seed=seed).select(range(num_samples))
print("Subset loaded.")

# setup a chain of transformations to apply to the images
transformation_chain = tv.transforms.Compose([
    # resize the image to 256x256 and take a center crop
    tv.transforms.Resize(int((256 / 224)) * extractor.size["height"]),
    tv.transforms.CenterCrop(extractor.size["height"]),
    # convert the image to a tensor
    tv.transforms.ToTensor(),
    # normalize the image
    tv.transforms.Normalize(extractor.image_mean, extractor.image_std),
])

# using this chain of transformations, we can setup a function to apply to each
# image in the dataset before we extract embeddings
def extract_embeddings(model: torch.nn.Module):
    device = model.device

    def preprocess(batch):
        images = batch["image"]
        transformed_image = torch.stack(
            [transformation_chain(image) for image in images]
        )
        new_batch = {"pixel_values": transformed_image.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        
        return {"embeddings": embeddings}
    
    return preprocess

# apply the transformation to the dataset
batch_size = 24
device = "cuda" if torch.cuda.is_available() else "cpu"
extract_fn = extract_embeddings(model.to(device))
candidate_subset_embeddings = candidate_image_subset.map(
    extract_fn,
    batched=True,
    batch_size=batch_size
)
print("Embeddings extracted.")

# after this, we create a list of identifiers for the candidate images for later
candidate_ids = []

for id in range(len(candidate_subset_embeddings)):
    label = candidate_image_subset[id]["labels"]
    entry = str(id) + "_" + str(label)
    candidate_ids.append(entry)

# gather the embeddings into a matrix
all_candidate_embeddings = np.array(candidate_subset_embeddings["embeddings"])
all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)
print("Embeddings gathered.")

# cosine similarity is used to compare embeddings
def compute_scores(embed_one, embed_two):
    scores = torch.nn.functional.cosine_similarity(embed_one, embed_two)
    return scores.numpy().tolist()

def fetch_similar_beans(image, top_k=5):
    image_transformed = transformation_chain(image).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(device)}

    with torch.no_grad():
        query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()

    sim_scores = compute_scores(query_embeddings, all_candidate_embeddings)
    similary_mapping = dict(zip(candidate_ids, sim_scores))

    sorted_similarities = dict(
        sorted(similary_mapping.items(), key=lambda item: item[1], reverse=True)
    )
    id_entries = list(sorted_similarities.keys())[:top_k]

    ids = list(map(lambda entry: int(entry.split("_")[0]), id_entries))
    labels = list(map(lambda entry: int(entry.split("_")[-1]), id_entries))
    return ids, labels

# we can now use this function to find similar images
def plot_images(images, labels):
    if not isinstance(images, list):
        labels = labels.tolist()

    plt.figure(figsize=(20, 6))
    columns = 6
    for i, image in enumerate(images):
        label_id = int(labels[i])
        ax = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        if i == 0:
            ax.set_title(f'Query Image\nLabel: {idx_to_labels[label_id]}')
        else:
            ax.set_title(f'Similar Image #{i}\nLabel: {idx_to_labels[label_id]}')
        plt.axis("off")
        plt.imshow(np.array(image).astype("int"))

    plt.show()

# now we can test it out
query_image_idx = np.random.choice(len(dataset["test"]))
query_image = dataset["test"][query_image_idx]["image"]
query_label = dataset["test"][query_image_idx]["labels"]
print(f"Query image index: {query_image_idx}")
print(f"Query image label: {query_label}")

similar_image_idxs, similar_image_labels = fetch_similar_beans(query_image)

similar_images = []
similar_labels = []

for id, label in zip(similar_image_idxs, similar_image_labels):
    similar_images.append(candidate_subset_embeddings[id]["image"])
    similar_labels.append(candidate_subset_embeddings[id]["labels"])

similar_images.insert(0, query_image)
similar_labels.insert(0, query_label)
plot_images(similar_images, similar_labels)
