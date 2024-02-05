import torch
import numpy as np

from transformers import ViTImageProcessor, TrainingArguments, Trainer, ViTForImageClassification
from datasets import load_dataset, load_metric

# we're gonna try to train a base ViT model on cifar10

model_checkpoint = 'google/vit-base-patch16-224'
model = None
processor = ViTImageProcessor.from_pretrained(model_checkpoint)

dataset = load_dataset('cifar10')

def transform_to_vit_input(batch):
    inputs = processor([x for x in batch['img']], return_tensors='pt')
    inputs['label'] = batch['label']
    return inputs

prepared_dataset = dataset.with_transform(transform_to_vit_input)
training_dataset = prepared_dataset['train']
testing_dataset = prepared_dataset['test']

def collate(batch):
    pixel_values = torch.stack([x['pixel_values'] for x in batch])
    labels = torch.tensor([x['label'] for x in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    return metric.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)

labels = dataset['train'].features['label'].names

model = ViTForImageClassification.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True,
    id2label={str(i) : c for i, c in enumerate(labels)},
    label2id={c : str(i) for i, c in enumerate(labels)}
)

model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=len(labels))

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    evaluation_strategy='steps',
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
    train_dataset=training_dataset,
    eval_dataset=testing_dataset,
    tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics('train', train_results.metrics)
trainer.save_metrics('train', train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
trainer.log_metrics('eval', metrics)
trainer.save_metrics('eval', metrics)