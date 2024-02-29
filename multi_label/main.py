import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

df = pd.read_csv('multilabel_modified/multilabel_classification(6)-reduced_modified.csv')

labels = list(df.columns[2:])
id2label = {i: label for i, label in enumerate(labels)}

# now we load the model
model_checkpoint = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    problem_type="multi_label_classification",
    id2label=id2label,
    ignore_mismatched_sizes=True
)

# creating a dataset
class MultilabelDataset(Dataset):
    def __init__(self, root, df, transform):
        self.root = root
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        item = self.df.iloc[index]
        image_path = os.path.join(self.root, item["Image_Name"])
        image = Image.open(image_path)
        pixel_values = self.transform(image)
        labels = item[2:].values.astype(np.float32)
        labels = torch.from_numpy(labels)

        return pixel_values, labels
    
    def __len__(self):
        return len(self.df)
    
    def __dict__(self):
        return self.df
    
size = image_processor.size["height"]
mean = image_processor.image_mean
std = image_processor.image_std

transform = Compose([
    Resize((size, size), interpolation=Image.BILINEAR),  # Specify the interpolation mode as a string
    ToTensor(),
    Normalize(mean, std)
])

train_dataset = MultilabelDataset("multilabel_modified/images", df, transform)

# verify
# pixel_values, labels = train_dataset[63]
# print(pixel_values.shape)

# unnormalized_image = (pixel_values.numpy() * np.array(std)[:, None, None]) + np.array(mean)[:, None, None]
# unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
# unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

# labels_strings = [id2label[label] for label in torch.nonzero(labels).squeeze().tolist()]

# plt.imshow(unnormalized_image)
# plt.title(labels_strings)
# plt.show()

# creating a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# training the model
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

trainer.train()