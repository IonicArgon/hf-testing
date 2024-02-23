import torch
import numpy as np
import albumentations as A

from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict, Metric
import evaluate

from typing import Any, Dict, List, Tuple, Optional
from LogClasses import WarningMessage, NoCheckpointException, NoTrainingDataException, NoTrainingArgumentException

class GenericModel:
    def __init__(
        self,
        model_checkpoint: str = None,
        dataset: DatasetDict = None,
        metric: str = "accuracy",
    ):
        if model_checkpoint is None:
            raise NoCheckpointException

        self.model_checkpoint: str = model_checkpoint
        self.processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
            self.model_checkpoint
        )
        self.model: Optional[AutoModelForImageClassification] = None
        self.dataset: DatasetDict = dataset
        self.metric: Metric = evaluate.load(metric)

        if self.dataset is None:
            print(WarningMessage("No dataset provided for training"))

    def _transform_to_model_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.processor([x for x in batch["image"]], return_tensors="pt")

        # process our labels
        possible_features = [label for label in batch.keys() if label != "image"]
        feature_labels = [batch.unique(feature) for feature in possible_features]
        mappings = {}
        for i, feature in enumerate(possible_features):
            mappings[feature] = {feature_label: j for j, feature_label in enumerate(feature_labels[i])}
        self.combinations = 1
        for feature in possible_features:
            self.combinations *= len(mappings[feature])

        labels = []
        for i in range(len(batch["image"])):
            label = 0
            for j, feature in enumerate(possible_features):
                label += mappings[feature][batch[feature][i]] * (len(mappings[feature]) ** j)
            labels.append(label)

        inputs["labels"] = torch.tensor(labels)
        return inputs

    def _augmentations(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        transformations = A.Compose(
            [
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.5,
                ),
                A.Rotate(limit=20, p=0.25),
                A.RandomBrightnessContrast(p=0.25),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.25
                ),
                A.Resize(224, 224),
                A.Normalize(),
            ]
        )

        batch["pixel_values"] = [
            transformations(image=np.array(x))["image"] for x in batch["image"]
        ]

        return batch

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.tensor([x["labels"] for x in batch])
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    def _compute_metrics(
        self, eval_pred: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        predictions, labels = eval_pred
        return self.metric.compute(predictions=predictions, references=labels)

    def train(self, training_args: TrainingArguments) -> None:
        if training_args is None:
            raise NoTrainingArgumentException

        if self.dataset is None:
            raise NoTrainingDataException
        
        # prepare data
        def augment_and_transform(batch):
            batch = self._augmentations(batch)
            batch = self._transform_to_model_input(batch)
            return batch
        
        prepared = self.dataset.with_transform(augment_and_transform)
        self.train_split = prepared["train"]
        self.valid_split = prepared["valid"]
        self.test_split = prepared["test"]

        labels = list(range(9))

        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(labels),
            ignore_mismatched_sizes=True,
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
        )

        self.model.classifier = torch.nn.Linear(
            self.model.classifier.in_features, len(labels)
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self._collate,
            compute_metrics=self._compute_metrics,
            train_dataset=self.train_split,
            eval_dataset=self.valid_split,
        )

        trainer.train()
        trainer.evaluate(self.test_split)
        trainer.save_model()
        trainer.save_metrics("training_metrics.json")

# if __name__ == "__main__":
#     try:
#         test = GenericModel()
#     except NoCheckpointException as e:
#         print(e)

#     try:
#         test = GenericModel("google/vit-base-patch16-224")
#         test.train(None)
#     except NoTrainingArgumentException as e:
#         print(e)

#     try:
#         test = GenericModel("google/vit-base-patch16-224")
#         training_args = TrainingArguments(
#             output_dir="./results",
#             per_device_train_batch_size=16,
#             evaluation_strategy="steps",
#             num_train_epochs=4,
#             fp16=True,
#             save_steps=100,
#             eval_steps=100,
#             logging_steps=10,
#             learning_rate=2e-4,
#             save_total_limit=2,
#             remove_unused_columns=False,
#             push_to_hub=False,
#             load_best_model_at_end=True,
#         )
#         test.train(training_args)
#     except NoTrainingDataException as e:
#         print(e)