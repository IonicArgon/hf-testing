import torch
import torch.nn.functional as F
import albumentations as A
import numpy as np

from Dataset import Dataset

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForImageClassification, AutoImageProcessor

if __name__ == "__main__":
    dataset = Dataset()
    hf_dataset = dataset.get_dataset()

    BEAR_TYPES = ["giant_panda", "grizzly_bear", "polar_bear"]
    ACTIVITY_TYPES = ["eating", "sitting", "standing"]
    LABEL_COLUMNS = BEAR_TYPES + ACTIVITY_TYPES

    MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

    feature_extractor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        problem_type="multi_label_classification",
        num_labels=len(LABEL_COLUMNS)
    )

    albumentations_transform = A.Compose([
        A.Resize(feature_extractor.size["width"], feature_extractor.size["height"]),
        A.RandomBrightnessContrast(p=0.2),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0)
        ], p=0.2),
        A.Normalize(
            mean=feature_extractor.image_mean,
            std=feature_extractor.image_std,
            p=1.0
        )
    ])

    def transforms(batch):
        batch["pixel_values"] = [
            albumentations_transform(image=np.array(image))["image"] for image in batch["image"]
        ]
        return batch
    
    def encode_onehots(batch):
        giant_panda = batch["giant_panda"]
        grizzly_bear = batch["grizzly_bear"]
        polar_bear = batch["polar_bear"]
        eating = batch["eating"]
        sitting = batch["sitting"]
        standing = batch["standing"]

        one_hots = [[0] * 6 for _ in range(len(giant_panda))]

        for i in range(len(one_hots)):
            one_hots[i][0] = giant_panda[i]
            one_hots[i][1] = grizzly_bear[i]
            one_hots[i][2] = polar_bear[i]
            one_hots[i][3] = eating[i]
            one_hots[i][4] = sitting[i]
            one_hots[i][5] = standing[i]

        batch["labels"] = torch.tensor(one_hots, dtype=torch.float32)
        return batch
    
    def collate_fn(batch):
        images = []
        labels = []
        for item in batch:
            image = np.moveaxis(item["pixel_values"], source=2, destination=0)
            images.append(torch.from_numpy(image))
            labels.append(torch.tensor(item["labels"], dtype=torch.float32))

        pixel_values = torch.stack(images)
        labels = torch.stack(labels)

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    
    processed_train_dataset = hf_dataset["train"]
    processed_valid_dataset = hf_dataset["valid"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_dataset = self.train_dataset.map(encode_onehots, batched=True)
            self.train_dataset = self.train_dataset.map(
                lambda batch: transforms(batch), batched=True
            )

            self.eval_dataset = self.eval_dataset.map(encode_onehots, batched=True)
            self.eval_dataset = self.eval_dataset.map(
                lambda batch: transforms(batch), batched=True
            )

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
        tokenizer=feature_extractor,
        data_collator=collate_fn
    )

    trainer.train()