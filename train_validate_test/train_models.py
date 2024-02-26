from GenericModel import GenericModel
from Dataset import Dataset

from transformers import TrainingArguments

from datasets import load_dataset, DatasetInfo

if __name__ == "__main__":
    dataset = Dataset()
    hf_dataset = dataset.get_dataset()

    # possible_features = [label for label in hf_dataset["train"].features if label != "image"]
    # feature_labels = [hf_dataset["train"].unique(feature) for feature in possible_features]
    # mappings = {}
    # for i, feature in enumerate(possible_features):
    #     mappings[feature] = {feature_label: j for j, feature_label in enumerate(feature_labels[i])}
    # print(mappings)

    # # grab the first data point
    # first_data_point = hf_dataset["train"][0]
    # print(first_data_point)

    # # we're gonna try to map the labels into a one dimensional array
    # # i.e. if it's giant_panda + eating, it looks like
    # # [1, 0, 0, 1, 0, 0] based on the mappings

    # unconcatenated_labels = []
    # for i, feature in enumerate(possible_features):
    #     unconcatenated_labels.append([0 for _ in range(len(mappings[feature]))])
    
    # for i, feature in enumerate(possible_features):
    #     unconcatenated_labels[i][mappings[feature][first_data_point[feature]]] = 1

    # concatenated_labels = []
    # for label in unconcatenated_labels:
    #     concatenated_labels.extend(label)

    # print(concatenated_labels)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
    )

    model = GenericModel(
        model_checkpoint="google/vit-base-patch16-224",
        dataset=hf_dataset,
        metric="accuracy",
    )

    model.train(training_args)
