from utils import seed_everything
from utils import logger
from utils import get_optimizer
from utils import get_optimizer_params
from utils import orthogonality_error
from utils import get_checkpoint_path
import argparse
import torch
from cil_dataloader import get_cifar100_cil_loaders
import clip
from GeoStack.GeoLayer import GeoLayer
from tqdm import tqdm
from losses import contrastive
from losses import convex_ortho_align

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model_name, num_tasks=4, geolayer=False, biclip=False):
    if not geolayer and not biclip:
        logger.error("No train setting provide. Set geolayer or biclip!")
        exit()
    elif geolayer and biclip:
        logger.error("Cannot train geolayer and biclip at the same time!")
        exit()
    if geolayer:
        logger.info(f"Training GeoLayer")
    else:
        logger.info(f"Training BiCLIP")

    logger.info(f"Using CLIP Backbone: {model_name}")
    clip_model, preprocess = clip.load(model_name, device=device, jit=False)
    clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad = False

    dataset_name = "cifar100"
    logger.info(f"Training on dataset: {dataset_name.upper()}.")
    train_loaders, _, prompt, classes = get_cifar100_cil_loaders(preprocess, num_tasks=num_tasks, num_shots=16)

    logger.info(f"Training on {num_tasks} tasks.")

    for task in range(num_tasks):
        logger.info("-"*20)
        logger.info(f"Task {task+1}/{num_tasks}")

        embed_dim = clip_model.visual.output_dim

        logger.info(f"Initializing GeoLayer with dims: {embed_dim} X {embed_dim} matrix.")
        model = GeoLayer(embed_dim, model_name).to(device)
        model.float()

        templates = [prompt % c for c in classes]
        text_tokens = clip.tokenize(templates).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features.to(device)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        optimizer_class = get_optimizer("adamw")

        lr = 8e-4
        weight_decay = 5e-2

        params = get_optimizer_params({"lr": lr, "weight_decay": weight_decay}, model)
        optimizer = optimizer_class(params)

        lambda_o = 0.99
        logger.info(f"Initializing GeoLayer with lambda: {lambda_o}")

        epochs = 40 if geolayer else 25
        train_loader = train_loaders[task]

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

            for images, labels in pbar:
                images = images.to(device).float()

                optimizer.zero_grad()

                text_features_this = text_features[labels]

                I_f = clip_model.encode_image(images).to(device)
                I_f = I_f / I_f.norm(dim=-1, keepdim=True)
                adapted_image = model(I_f)

                logit_scale = clip_model.logit_scale.exp()
                logits_per_image = logit_scale * adapted_image @ text_features_this.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(len(images), device=device)
                if geolayer:
                    loss = convex_ortho_align(logits_per_image, logits_per_text, ground_truth, model.W, lambda_o)
                else:
                    loss = contrastive(logits_per_image, logits_per_text, ground_truth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "W_norm": f"{model.W.norm().item():.2f}"
                })

                train_loss += loss.item()

                loss_info = {"loss": f"{loss.item():.4f}",
                             "W_norm": f"{model.W.norm().item():.2f}",
                             "W_ortho": orthogonality_error(model.W).item()
                             }

                pbar.set_postfix(loss_info)

        logger.info("Training complete.")
        logger.info(f"Orthogonality error: {orthogonality_error(model.W).item()}")

        checkpoint_path = get_checkpoint_path(dataset_name, geolayer=geolayer, biclip=biclip, scenario="cil", task=task, total_tasks=num_tasks)
        logger.info(f"Checkpoint path: {checkpoint_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'clip_model': model_name,
        }, checkpoint_path)


if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser(description="A script to process a dataset file.")

    parser.add_argument('-g', '--geo_layer', action='store_true',
                        help='Training geolayer expert.')
    parser.add_argument('-b', '--biclip', action='store_true',
                        help='Training BiCLIP model.')
    parser.add_argument('-n', '--num_tasks', type=int, default=4,
                        help='Number of tasks to train default(4).')

    args = parser.parse_args()

    model_name = "ViT-B/16"
    train(model_name, num_tasks=args.num_tasks, geolayer=args.geo_layer, biclip=args.biclip)