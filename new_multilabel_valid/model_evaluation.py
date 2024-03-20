import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pprint
import json
from torch.nn import Softmax

from Dataset import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

CHECKPOINT_LOCATIONS = [
    "./results/google-vit-base-patch16-224-in21k-training/checkpoint-2500",
    "./results/WinKawaks-vit-small-patch16-224-training/checkpoint-2500",
    "./results/WinKawaks-vit-tiny-patch16-224-training/checkpoint-2500",
]

ds_source = Dataset().get_dataset()

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
    seed = int(round(time.time()))
    random.seed(seed)
    THRESHOLD = 0.225
    SAMPLES = 150

    bear_types = ["giant_panda", "grizzly_bear", "polar_bear"]
    activity_types = ["eating", "sitting", "standing"]

    # test each checkpoint on a random batch of SAMPLES images
    # and create a confusion matrix for each
    # confusion matrix will be a 2x2 matrix
    # - TP: predicted and actual label match
    # - FP: predicted label is present, but actual label is not
    # - FN: actual label is present, but predicted label is not
    # - TN: predicted and actual label are both not present

    confusion_matrices = {}
    confusion_metrics = {}
    confidences = {}

    for checkpoint in CHECKPOINT_LOCATIONS:
        print(f"Testing checkpoint: {checkpoint}")

        feature_extractor = AutoImageProcessor.from_pretrained(checkpoint)
        model = AutoModelForImageClassification.from_pretrained(checkpoint)

        selected = random.sample(range(len(ds_source["test"])), SAMPLES)
        confusion_matrix = np.zeros((2, 2))
        img_count = 0

        # to calculate confidence of each label, store the probabilities
        # for each label in a dictionary
        confidences[checkpoint] = {
            "giant_panda": {
                "mean": 0,
                "std": 0,
                "count": 0
            },
            "grizzly_bear": {
                "mean": 0,
                "std": 0,
                "count": 0
            },
            "polar_bear": {
                "mean": 0,
                "std": 0,
                "count": 0
            },
            "eating": {
                "mean": 0,
                "std": 0,
                "count": 0
            },
            "sitting": {
                "mean": 0,
                "std": 0,
                "count": 0
            },
            "standing": {
                "mean": 0,
                "std": 0,
                "count": 0
            }
        }

        for idx in selected:
            img = ds_source["test"][idx]["image"]
            labels = ds_source["test"][idx]["labels"]
            labels = labels.split(" ")
            inputs = feature_extractor(images=np.array(img), return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits

            predicted_labels, probabilities = get_predicted_labels(logits, THRESHOLD)

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0

            # calculate true and false positives
            for label in predicted_labels[0]:
                if label in labels:
                    true_positives += 0.5
                else:
                    false_positives += 0.5

            # calculate false negatives
            for label in labels:
                if label not in predicted_labels[0]:
                    false_negatives += 0.5

            # calculate true negatives
            for bear in bear_types:
                if bear not in labels and bear not in predicted_labels[0]:
                    true_negatives += 0.5
            for activity in activity_types:
                if activity not in labels and activity not in predicted_labels[0]:
                    true_negatives += 0.5

            confusion_matrix[0, 0] += true_positives
            confusion_matrix[0, 1] += false_positives
            confusion_matrix[1, 0] += false_negatives
            confusion_matrix[1, 1] += true_negatives

            # do a running average of the confidence for each label
            for i, label in enumerate(confidences[checkpoint]):
                confidences[checkpoint][label]["count"] += 1
                confidences[checkpoint][label]["mean"] += (probabilities[0][i].item() - confidences[checkpoint][label]["mean"]) / confidences[checkpoint][label]["count"]
                confidences[checkpoint][label]["std"] += (probabilities[0][i].item() - confidences[checkpoint][label]["mean"]) * (probabilities[0][i].item() - confidences[checkpoint][label]["mean"])

            img_count += 1
            print(f"{' ' * 30}", end="\r")
            print(f"Processed {img_count} images", end="\r")

        confusion_matrices[checkpoint] = confusion_matrix

    # now calculate confusion metrics
    for checkpoint, matrix in confusion_matrices.items():
        tp = matrix[0, 0]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        tn = matrix[1, 1]

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        fallout = fp / (fp + tn)
        miss_rate = fn / (fn + tp)

        confusion_metrics[checkpoint] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "fallout": fallout,
            "miss_rate": miss_rate
        }

    pprint.pprint(confusion_metrics)
    pprint.pprint(confidences)

    # save to json files

    # confusio matrices are numpy arrays, so convert them to lists
    matrices_for_json = {}
    for checkpoint, matrix in confusion_matrices.items():
        matrices_for_json[checkpoint] = matrix.tolist()

    with open("confusion_matrices.json", "w") as f:
        json.dump(matrices_for_json, f)
    
    with open("confusion_metrics.json", "w") as f:
        json.dump(confusion_metrics, f)

    with open("confidences.json", "w") as f:
        json.dump(confidences, f)

    # now plot the confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, (checkpoint, matrix) in enumerate(confusion_matrices.items()):
        ax[i].imshow(matrix, cmap="Blues")
        ax[i].set_xticks([0, 1])
        ax[i].set_yticks([0, 1])
        ax[i].set_xticklabels(["Positive", "Negative"])
        ax[i].set_yticklabels(["Positive", "Negative"])
        ax[i].set_xlabel("Predicted")
        ax[i].set_ylabel("Actual")
        ax[i].set_title(f"Confusion Matrix for {checkpoint[10:-25]}")

        for j in range(2):
            for k in range(2):
                ax[i].text(k, j, f"{matrix[j, k]}", ha="center", va="center", color="red")

    plt.tight_layout()
    plt.show()
            