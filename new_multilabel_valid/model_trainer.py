import torch
import torch.nn.functional as F

from Dataset import Dataset

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForImageClassification, AutoImageProcessor

if __name__ == "__main__":
    dataset = Dataset()
    hf_dataset = dataset.get_dataset()

    bear_types = ["giant_panda", "grizzly_bear", "polar_bear"]
    activity_types = ["eating", "sitting", "standing"]

    model_checkpoint = "google/vit-base-patch16-224-in21k"
    feature_extractor = AutoImageProcessor.from_pretrained(model_checkpoint)

    label_columns = bear_types + activity_types

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        problem_type="multi_label_classification",
        num_labels=len(label_columns)
    )

    def preprocess(batch):
        batch["pixel_values"] = feature_extractor(batch["image"], return_tensors="pt")["pixel_values"]

        # need to process the columns into a one-hot encoded tensor
        giant_panda = batch.pop("giant_panda")
        grizzly_bear = batch.pop("grizzly_bear")
        polar_bear = batch.pop("polar_bear")
        eating = batch.pop("eating")
        sitting = batch.pop("sitting")
        standing = batch.pop("standing")

        one_hots = [[0] * 6 for _ in range(len(batch["image"]))]

        for i in range(len(batch["image"])):
            one_hots[i][0] = giant_panda[i]
            one_hots[i][1] = grizzly_bear[i]
            one_hots[i][2] = polar_bear[i]
            one_hots[i][3] = eating[i]
            one_hots[i][4] = sitting[i]
            one_hots[i][5] = standing[i]

        batch["labels"] = torch.tensor(one_hots, dtype=torch.float32)
        return batch
    
    processed_train_dataset = hf_dataset["train"].map(preprocess, batched=True)
    processed_valid_dataset = hf_dataset["valid"].map(preprocess, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return (loss, outputs) if return_outputs else loss
        
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_valid_dataset,
    )

    trainer.train()