import random
import numpy as np
import torch
import os
import yaml
from torch.utils.data import Dataset
from rich.logging import RichHandler
import logging
import csv

MDA_MODEL_DATA = "Data/MDA"
CIL_MODEL_DATA = "Data/CIL"

CONFIG_FOLDER = "configs"

def get_logger():
    # Configure logging to use RichHandler
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

    return logging.getLogger("rich")

logger = get_logger()

def write_results(data, file_name):
    keys = data[0].keys()
    # Check if file exists and has content
    file_empty = not os.path.exists(file_name) or os.path.getsize(file_name) == 0

    # 2. Write to the file
    with open(file_name, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        if file_empty:
            dict_writer.writeheader()
        dict_writer.writerows(data)

def get_checkpoint_path(dataset_name, scenario="mda", geolayer=False, biclip=False, task=None, total_tasks=None):
    assert scenario in ["mda", "cil"], "Unknown scenario"
    if scenario == "mda":
        folder = MDA_MODEL_DATA
        if geolayer:
            file_name = f"{dataset_name}_geolayer.pth"
        elif biclip:
            file_name = f"{dataset_name}_biclip.pth"
        else:
            logger.error(f"Unknown setting!")
            exit()
    elif scenario == "cil":
        assert task is not None, "task is required"
        assert total_tasks is not None, "total_tasks is required"
        assert task < total_tasks, f"task is out of range total_tasks={total_tasks}, task={task}"
        assert total_tasks > 1, "total_tasks should be > 1"
        folder = CIL_MODEL_DATA
        if geolayer:
            file_name = f"{total_tasks}_{dataset_name}_geolayer_task_{task}.pth"
        elif biclip:
            file_name = f"{total_tasks}_{dataset_name}_biclip_task_{task}.pth"
        else:
            logger.error(f"Unknown setting!")
            exit()

    return os.path.join(folder, file_name)

def seed_everything(seed=42):
    print("seeding Everything!")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config_file(dataset, backbone="vit16", geolayer=True):

    assert backbone.lower() in ["vit16"], f"backbone {backbone} not supported"

    config_file = os.path.join(CONFIG_FOLDER, f"{dataset}.yml")
    logger.info(f"Loading config from {config_file}")
    cfg = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    config_name = f"{backbone.lower()}"
    if geolayer:
        config_name += "_geolayer"

    logger.info(f"Loading config from {config_name}.")
    return cfg[config_name]

def get_optimizer(name):
    logger.info(f"Optimizer Name: {name}")
    if name.lower() == "adam":
        return torch.optim.Adam
    elif name.lower() == "adamw":
        return torch.optim.AdamW
    elif name.lower() == "sgd":
        return torch.optim.SGD

def get_optimizer_params(training_params, model):
    lr = float(training_params["lr"])
    weight_decay = float(training_params["weight_decay"])
    logger.info(f"W - lr: {lr}, weight_decay: {weight_decay}")
    params = [{'params': model.W, 'lr': lr, 'weight_decay': weight_decay}]

    logger.info(f"Trainable Variable Count: {len(params)}")
    return params

def get_scheduler(training_prams, optimizer, epochs):
    if training_prams["lr_scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif training_prams["lr_scheduler"] == "cosine+warmup":
        logger.info("Custom Scheduler: Linear Warmup (5epochs) + Cosine Annealing")
        import math
        def get_lr_lambda(epoch):
            warmup_epochs = 10
            total_epochs = epochs
            assert warmup_epochs <= total_epochs, "Scheduler Error"

            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            # 2. Cosine Decay Phase
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
        return scheduler
    elif training_prams["lr_scheduler"] == "steplr":
        logger.info(f"Using StepLR Scheduler")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return scheduler

def orthogonality_error(W):
    D = W.shape[0]
    return torch.norm(torch.mm(W.t(), W) - torch.eye(D, device=W.device)) / D


def get_stack(stack):
    datasets = ["imagenet",
                "caltech101",
                "flowers101",
                "food101",
                "eurosat",
                "dtd"
                ]

    split_stack = stack.split("->")

    dataset_list = []
    get_match = lambda lst, p: [s for s in lst if s.startswith(p)]

    for p in split_stack:
        matches = get_match(datasets, p)
        assert len(matches) == 1, f"{p} matches more than 1 dataset!{matches}, try using first 2-3 characters"
        dataset_list.append(matches[0])

    return dataset_list

class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]

        if hasattr(x, 'convert'):
            x = x.convert("RGB")

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
