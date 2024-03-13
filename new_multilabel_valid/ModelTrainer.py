import torch
import torch.nn.functional as F
import albumentations as A
import numpy as np

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForImageClassification, AutoImageProcessor

class TrainerForMultiLabelClassification(Trainer):
    def __init__(self, one_hot_encoder, transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.one_hot_encoder = one_hot_encoder
        self.transformer = transformer

        self.train_dataset = self.train_dataset.map(self.transformer, batched=True)
        self.train_dataset = self.train_dataset.map(self.one_hot_encoder, batched=True)

        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.map(self.transformer, batched=True)
            self.eval_dataset = self.eval_dataset.map(self.one_hot_encoder, batched=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss

class ModelTrainer:
    def __init__(self, hf_dataset, model_checkpoint, model_params):
        self.hf_dataset = hf_dataset
        self.model_checkpoint = model_checkpoint
        self.model_params = model_params

        self.image_processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint,
            problem_type=self.model_params["problem_type"],
            num_labels=self.model_params["num_labels"],
            ignore_mismatched_sizes=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_transforms(self, batch):
        transforms = A.Compose([
            A.Resize(self.image_processor.size["width"], self.image_processor.size["height"]),
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussNoise(var_limit=(5.0, 30.0), p=1.0)
            ], p=0.2),
            A.Normalize(
                mean=self.image_processor.image_mean if self.image_processor.image_mean else [0.485, 0.456, 0.406],
                std=self.image_processor.image_std if self.image_processor.image_std else [0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            )
        ])

        batch["pixel_values"] = [
            transforms(image=np.array(image))["image"] for image in batch["image"]
        ]
        return batch
    
    def _encode_onehots(self, batch):
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
    
    def _collate(self, batch):
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
    
    def train(self, training_args):
        self.model.to(self.device)

        model_name = self.model_checkpoint.replace("/", "-")

        training_args = TrainingArguments(
            output_dir=f"./results/{model_name}-training",
            num_train_epochs=training_args["num_train_epochs"],
            per_device_train_batch_size=training_args["per_device_train_batch_size"],
            per_device_eval_batch_size=training_args["per_device_eval_batch_size"],
            warmup_steps=training_args["warmup_steps"],
            weight_decay=training_args["weight_decay"],
            logging_dir=f"./{model_name}-logs",
            logging_steps=training_args["logging_steps"],
            evaluation_strategy=training_args["evaluation_strategy"],
        )

        trainer = TrainerForMultiLabelClassification(
            one_hot_encoder=self._encode_onehots,
            transformer=self._apply_transforms,
            model=self.model,
            args=training_args,
            train_dataset=self.hf_dataset["train"],
            eval_dataset=self.hf_dataset["valid"],
            data_collator=self._collate,
            tokenizer=self.image_processor,
        )

        train_results = trainer.train()
        trainer.save_model(f"./models/{model_name}")
        trainer.save_metrics("train", train_results.metrics)
        trainer.log_metrics("train", train_results.metrics)

# if __name__ == "__main__":
#     from Dataset import Dataset

#     dataset = Dataset()
#     hf_dataset = dataset.get_dataset()

#     model = ModelTrainer(
#         hf_dataset=hf_dataset,
#         model_checkpoint="google/vit-base-patch16-224-in21k",
#         model_params={"problem_type": "multi_label_classification", "num_labels": 6}
#     )

#     training_args = {
#         "num_train_epochs": 5,
#         "per_device_train_batch_size": 8,
#         "per_device_eval_batch_size": 8,
#         "warmup_steps": 500,
#         "weight_decay": 0.01,
#         "logging_steps": 10,
#         "evaluation_strategy": "epoch"
#     }

#     model.train(training_args)
