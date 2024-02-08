from datasets import load_dataset
from transformers import DeiTImageProcessor, DeiTForImageClassification

model_ckpt = "facebook/deit-base-distilled-patch16-224"
model = DeiTForImageClassification.from_pretrained("./results")
processor = DeiTImageProcessor.from_pretrained(model_ckpt)

# pick 5x5 random images from the test set
ds_source = load_dataset("imagefolder", data_dir="training_data")
ds_test = ds_source["test"]

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Softmax

seed = int(round(time.time()))
random.seed(seed)

selected = random.sample(range(len(ds_test)), 25)

fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i, idx in enumerate(selected):
    img = ds_test[idx]["image"]
    label = ds_test[idx]["label"]
    inputs = processor(images=np.array(img), return_tensors="pt")
    outputs = model(**inputs)
    predicted = np.argmax(outputs.logits.detach().numpy())
    logits = outputs.logits

    sm = Softmax(dim=1)
    confidence = sm(logits)
    confidence = confidence[0][predicted]

    label = "cat" if label == 0 else "dog"
    predicted = "cat" if predicted == 0 else "dog"

    ax[i//5, i%5].imshow(img)
    ax[i//5, i%5].set_title(f"Predicted: {predicted}\nActual: {label}\nConfidence: {confidence:.2f}")
    ax[i//5, i%5].axis("off")

plt.tight_layout()
plt.show()