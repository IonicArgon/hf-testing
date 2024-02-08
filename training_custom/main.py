import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import time
import random

from datasets import load_dataset, load_metric
from transformers import DeiTImageProcessor, DeiTForImageClassification, TrainingArguments, Trainer

# seed the random number generator
seed = int(round(time.time()))
random.seed(seed)
print(f"Random seed: {seed}")

model_ckpt = "facebook/deit-base-distilled-patch16-224"
model = None
processor = DeiTImageProcessor.from_pretrained(model_ckpt)

ds_source = load_dataset("imagefolder", data_dir="training_data")

#? note for myself: 0 is cat, 1 is dog

def transform_to_deit_input(batch):
    inputs = processor([x for x in batch["image"]], return_tensors="pt")
    inputs["label"] = batch["label"]
    return inputs

def augmentations(batch):
    transformations = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], p=0.5),
        A.Rotate(limit=20, p=0.25),
        A.RandomBrightnessContrast(p=0.25),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.25
        ),
        A.Resize(224, 224),
    ])

    batch["pixel_values"] = [
        transformations(image=np.array(x))["image"] for x in batch["image"]
    ]

    return batch

def augment_and_transform(batch):
    batch = augmentations(batch)
    batch = transform_to_deit_input(batch)
    return batch

ds_prepared = ds_source.with_transform(augment_and_transform)
ds_train = ds_prepared["train"]
ds_test = ds_prepared["test"]

def collate(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    return metric.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)

labels = ds_source["train"].features["label"].names

model = DeiTForImageClassification.from_pretrained(
    model_ckpt,
    num_labels=len(labels),
    ignore_mismatched_sizes=True,
    id2label={str(i) : c for i, c in enumerate(labels)},
    label2id={c : str(i) for i, c in enumerate(labels)}
)

model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=len(labels))

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate,
    compute_metrics=compute_metrics,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    tokenizer=processor,
)

print("Training with arguments:")
print(training_args)
print("Size of training set:", len(ds_train))

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

print("Training complete.")

# now try and use the model to predict the label of a new image
# load the model
model = DeiTForImageClassification.from_pretrained("./results")
processor = DeiTImageProcessor.from_pretrained(model_ckpt)

# pick a random image from the test set
index = random.randint(0, len(ds_source["test"]) - 1)
image = ds_source["test"][index]["image"]
label = ds_source["test"][index]["label"]
label = "cat" if label == 0 else "dog"

# display the image
plt.imshow(image)
plt.title(f"True label: {label}")
plt.axis("off")
plt.show(block=False)

# transform the image to the format expected by the model
inputs = processor(images=image, return_tensors="pt")

# make a prediction
outputs = model(**inputs)

# get the predicted label
predicted_label = labels[outputs.logits.argmax().item()]

# display the predicted label
print(f"Predicted label: {predicted_label}")
input("Press Enter to continue...")
