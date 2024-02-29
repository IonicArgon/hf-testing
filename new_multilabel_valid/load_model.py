import time
import random
import numpy as np
import matplotlib.pyplot as plt

from Dataset import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.nn import Softmax


def get_predicted_labels(logits, threshold=0.275):
    sm = Softmax(dim=1)
    probabilities = sm(logits)
    predicted_labels = []
    labels = [
        "giant_panda",
        "grizzly_bear",
        "polar_bear",
        "eating",
        "sitting",
        "standing",
    ]

    for prob in probabilities:
        current_labels = [labels[i] for i in range(len(prob)) if prob[i] > threshold]
        predicted_labels.append(current_labels)

    return predicted_labels, probabilities


if __name__ == "__main__":
    dataset = Dataset()
    hf_dataset = dataset.get_dataset()

    bear_types = ["giant_panda", "grizzly_bear", "polar_bear"]
    activity_types = ["eating", "sitting", "standing"]

    model_checkpoint = "google/vit-base-patch16-224-in21k"
    feature_extractor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        "./results/checkpoint-2500",
        problem_type="multi_label_classification",
        num_labels=len(bear_types) + len(activity_types),
    )

    seed = int(round(time.time()))
    random.seed(seed)

    selected = random.sample(range(len(hf_dataset["test"])), 5)

    # for each image
    # subplot 1: image, title is actual and predicted labels
    # subplot 2: bar chart of probabilities

    fig, ax = plt.subplots(5, 2, figsize=(10, 10))
    for i, idx in enumerate(selected):
        img = hf_dataset["test"][idx]["image"]
        labels = hf_dataset["test"][idx]["labels"]

        inputs = feature_extractor(images=np.array(img), return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        predicted_labels, probabilities = get_predicted_labels(logits)
        predicted_labels = " ".join(predicted_labels[0])

        ax[i, 0].imshow(img)
        ax[i, 0].set_title(f"Actual: {labels}\nPredicted: {predicted_labels}")
        ax[i, 0].axis("off")

        ax[i, 1].barh(
            range(len(probabilities[0])),
            probabilities[0].detach().numpy(),
            tick_label=["giant_panda", "grizzly_bear", "polar_bear", "eating", "sitting", "standing"],
        )
        ax[i, 1].axvline(0.275, color="red", linestyle="--")
        ax[i, 1].bar_label(ax[i, 1].containers[0], fmt="%.3f")
        ax[i, 1].set_xlim(0, 1)
        ax[i, 1].set_title("Probabilities")

    plt.tight_layout()
    plt.show()

