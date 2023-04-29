import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification


device = "cuda:0"

image_processor = ViTImageProcessor.from_pretrained('model_CRC02_10epochs/')
model = ViTForImageClassification.from_pretrained('model_CRC02_10epochs/').to(device)


size = (image_processor.size["height"], image_processor.size["width"])
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)


def preprocess_val(example_batch):
    """Apply _val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
# DataLoaders creation:
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values.to(device), "labels": labels}


dataset = load_dataset('imagefolder', data_files='CRC02-test-set/**', task='image-classification')
eval_dataset = dataset['train'].with_transform(preprocess_val)
eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=32)


import numpy as np

ppp = np.array([])
rrr = np.array([])

for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    ppp = np.concatenate([ppp, predictions.cpu()])
    rrr = np.concatenate([rrr, batch["labels"]])
    if step > 4:
        break

id2label = {
    str(i): name
    for i, name in 
    enumerate(dataset["train"].features["labels"].names)
}

model_id2label = {
    v: k
    for k, v in model.config.label2id.items()
}
gt_mapped = [id2label[str(int(l))] for l in rrr]
pred_mapped = [model_id2label[str(int(l))] for l in ppp]

import sklearn.metrics

print(
    'accuracy:',
    sklearn.metrics.accuracy_score(gt_mapped, pred_mapped)
)

cfm_display = sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(gt_mapped, pred_mapped, labels=list(model_id2label.values())))
cfm_plot = cfm_display.plot()
ax = cfm_plot.ax_
ax.set_xticklabels(list(model_id2label.values()))
ax.set_yticklabels(list(model_id2label.values()))