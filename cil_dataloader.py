import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms, datasets
import os
from utils import ApplyTransform

def get_cifar100_cil_loaders(preprocess, batch_size=64, num_tasks=4, num_shots=-1):

    root = os.path.expanduser("~/.cache")

    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    full_train_ds = datasets.CIFAR100(root, train=True, download=True)
    full_test_ds = datasets.CIFAR100(root, train=False, download=True)
    train_targets = np.array(full_train_ds.targets)
    test_targets = np.array(full_test_ds.targets)

    classes = full_train_ds.classes
    clean_classes = [c.replace("_", " ").replace("-", " ").lower() for c in classes]
    prompt = "a photo of a %s, a type of object."

    classes_per_task = 100 // num_tasks
    train_loaders = []
    test_loaders = []


    for i in range(num_tasks):

        start_cls = i * classes_per_task
        end_cls = (i + 1) * classes_per_task
        task_classes = np.arange(start_cls, end_cls)

        task_train_indices = np.where(np.isin(train_targets, task_classes))[0]

        if num_shots > 0:
            few_shot_indices = []
            for c in task_classes:

                cls_indices = np.where(train_targets == c)[0]
                sampled = np.random.choice(cls_indices, num_shots, replace=False)
                few_shot_indices.extend(sampled)
            task_train_indices = few_shot_indices

        task_train_ds = Subset(full_train_ds, task_train_indices)

        task_train_ds = ApplyTransform(task_train_ds, transform=train_preprocess)
        train_loaders.append(DataLoader(task_train_ds, batch_size=batch_size, shuffle=True))

        task_test_indices = np.where(np.isin(test_targets, np.arange(0, end_cls)))[0]
        task_test_ds = Subset(full_test_ds, task_test_indices)
        task_test_ds = ApplyTransform(task_test_ds, transform=preprocess)
        test_loaders.append(DataLoader(task_test_ds, batch_size=batch_size, shuffle=False))

    return train_loaders, test_loaders, prompt, clean_classes
