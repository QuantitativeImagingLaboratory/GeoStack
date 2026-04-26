import argparse
from utils import seed_everything
from utils import get_config_file
from utils import logger
from utils import get_checkpoint_path
from utils import get_optimizer
from utils import get_optimizer_params
from utils import get_scheduler
from utils import orthogonality_error
from losses import contrastive
from losses import convex_ortho_align
import torch
import clip
from GeoStack.GeoLayer import GeoLayer
from mda_dataloader import get_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(config, geolayer=False, biclip=False, lambda_input=None):
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

    Dataset = config["Dataset"]
    Model = config["Model"]
    Training = config["Training"]

    vit_model_name = Model["backbone"]
    dataset = Dataset["dataset"]

    logger.info(f"Training on dataset: {dataset}")

    logger.info(f"Using CLIP Backbone: {vit_model_name}")
    clip_model, preprocess = clip.load(vit_model_name, device=device, jit=False)
    clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad = False

    embed_dim = clip_model.visual.output_dim

    logger.info(f"Initializing GeoLayer with dims: {embed_dim} X {embed_dim} matrix.")
    model = GeoLayer(embed_dim, vit_model_name).to(device)
    model.float()

    num_shot = Dataset["n_shot"]
    logger.info(f"Training on {num_shot}-shots protocol.")
    train_loader, _, prompt, classes = get_dataset(dataset, preprocess, num_shots=num_shot)

    # Creating text features for all class/prompts.
    templates = [prompt % c for c in classes]
    text_tokens = clip.tokenize(templates).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features.to(device)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    optimizer_class = get_optimizer(Training["optimizer"])
    params = get_optimizer_params(Training, model)
    if Training["optimizer"] == "sgd":
        optimizer = optimizer_class(params, momentum=0.9)
    else:
        optimizer = optimizer_class(params)

    epochs = Training["epochs"]
    if "lr_scheduler" in Training:
        logger.info("Using Scheduler")
        scheduler = get_scheduler(Training, optimizer, epochs)

    loss_function = Training["loss"]
    logger.info(f"Using Loss: {loss_function}")
    if loss_function == "contrastive+orthogonality":
        lambda_o = Training["lambda_ortho"] if lambda_input is None else lambda_input
        logger.info(f"Using Contrastive+Orthogonality, Lambda_ortho: {lambda_o}")

    checkpoint = get_checkpoint_path(dataset, geolayer=geolayer, biclip=biclip)
    logger.info(f"Saving to checkpoint: {checkpoint}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for images, labels in pbar:
            images = images.to(device).float()

            optimizer.zero_grad()

            I_f = clip_model.encode_image(images).to(device)
            I_f = I_f / I_f.norm(dim=-1, keepdim=True)
            adapted_image = model(I_f)

            text_features_batch = text_features[labels]

            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * adapted_image @ text_features_batch.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images), device=device)

            if loss_function == "contrastive":
                loss = contrastive(logits_per_image, logits_per_text, ground_truth)
            elif loss_function == "contrastive+orthogonality":
                loss = convex_ortho_align(logits_per_image, logits_per_text, ground_truth, model.W, lambda_o)

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

        if "lr_scheduler" in Training:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1} complete. Current LR: {current_lr:.6f}")
            scheduler.step()

    print("Training complete.")
    print(f"Orthogonality error: {orthogonality_error(model.W).item()}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'clip_model': vit_model_name,
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint)


if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser(description="A script to process a dataset file.")

    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name.')
    parser.add_argument('-g', '--geo_layer', action='store_true',
                        help='Training geolayer expert.')
    parser.add_argument('-b', '--biclip', action='store_true',
                        help='Training BiCLIP model.')

    args = parser.parse_args()

    cfg = get_config_file(args.dataset, geolayer=args.geo_layer)
    train(cfg, geolayer=args.geo_layer, biclip=args.biclip)