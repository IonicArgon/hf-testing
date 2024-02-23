from GenericModel import GenericModel
from Dataset import Dataset

from transformers import TrainingArguments

from datasets import load_dataset, DatasetInfo

if __name__ == "__main__":
    dataset = Dataset()
    hf_dataset = dataset.get_dataset()

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
