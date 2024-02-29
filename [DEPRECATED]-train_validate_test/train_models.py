from GenericModel import GenericModel
from Dataset import Dataset

# from transformers import TrainingArguments

if __name__ == "__main__":
    dataset = Dataset()
    hf_dataset = dataset.get_dataset()

    bear_types = ["giant_panda", "grizzly_bear", "polar_bear"]
    activity_types = ["eating", "sitting", "standing"]

    from datasets import ClassLabel
    hf_dataset = hf_dataset.class_encode_column("bear", ClassLabel(names=bear_types))
    hf_dataset = hf_dataset.class_encode_column("activity", ClassLabel(names=activity_types))

    #? works up to here

    # gonna try to work out the logic here
    from transformers import AutoTokenizer, AutoModelForImageClassification
    from transformers import TrainingArguments, Trainer

    model_checkpoint = "google/vit-base-patch16-224-in21k"
    feature_extractor = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        problem_type="multi_label_classification",
        num_labels=(len(bear_types) + len(activity_types))
    )
    processed_dataset = hf_dataset

    processed_dataset["train"] = processed_dataset["train"].select(range(100))

    training_args = TrainingArguments(
        output_dir="./results/test",
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    print("Batch Size:", training_args.per_device_train_batch_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=feature_extractor,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["valid"],
    )

    trainer.train()
