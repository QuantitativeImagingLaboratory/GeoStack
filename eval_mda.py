from utils import seed_everything
from utils import logger
from utils import get_stack
from utils import get_checkpoint_path
from MDADataLoader import get_dataset
import clip
from tqdm import tqdm
import argparse
from GeoStack.GeoStack import GeoStackCLIP
from GeoStack.GeoLayer import GeoLayer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_dataset(eval_model, dataset, num_shots=16, device=None):
    train_loader, test_loader, prompt, classes = get_dataset(dataset, eval_model.preprocess, num_shots=num_shots)

    templates = [prompt % c for c in classes]
    text_tokens = clip.tokenize(templates).to(device)
    text_features = eval_model.clip.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    eval_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        try:
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(device).float()
                I_f = eval_model(images)

                labels = labels.to(device)

                logits = eval_model.clip.logit_scale.exp() * I_f @ text_features.t()

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)

    return 100 * correct / total

def get_layer(dataset, vit_model_name="ViT-B/16", device=None, geo_layer=True, biclip=False):
    checkpoint_path = get_checkpoint_path(dataset, scenario="mda", geolayer=geo_layer, biclip=biclip)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if vit_model_name == "ViT-B/16":
        embed_dim = 512
    else:
        logger.error("ViT model not available")
        exit()

    model = GeoLayer(embed_dim=embed_dim, clip_model=vit_model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.float()
    model.eval()

    return model

def eval(stack, geolayer, biclip):
    backbone = "ViT-B/16"
    zero_shot = False
    if not geolayer and not biclip:
        zero_shot = True
        logger.warn("No eval setting provide. Evaluating zeroshot!")

    elif geolayer and biclip:
        logger.error("Cannot eval geolayer and biclip at the same time!")
        exit()
    if geolayer:
        logger.info(f"Evaluating GeoLayer")
    else:
        logger.info(f"Evaluating BiCLIP")

    dataset_stack = get_stack(stack)

    if zero_shot:
        logger.info(f"Zero-shot evaluation")
        model = GeoStackCLIP().to(device)
        model.float()
        model.eval()
    elif geolayer:
        logger.info("Evaluating based on GeoLayer Weights")
        all_geo_layers = []
        for k in dataset_stack:
            all_geo_layers.append(get_layer(k, geo_layer=geolayer, biclip=biclip))
        model = GeoStackCLIP(clip_model=backbone, geo_layers=all_geo_layers).to(device)
        model.eval()
    elif biclip:
        logger.info("Evaluating based on BiCLIP Weights")
        all_geo_layers = []
        for k in dataset_stack:
            all_geo_layers.append(get_layer(k, geo_layer=geolayer, biclip=biclip))
        model = GeoStackCLIP(clip_model=backbone, geo_layers=all_geo_layers).to(device)
        model.eval()

    accuracies = []

    for ind, k in enumerate(dataset_stack):
        logger.info(f"Evaluating {k} ({ind}/{len(dataset_stack)})")
        logger.info("-" * 20)
        logger.info(f"Evaluating on {k}")
        accuracies.append(evaluate_dataset(model, k, device=device))
        # accuracies.append(10)

        logger.info(f"Accuracy: {accuracies[-1]}")
        logger.info("-" * 20)

if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser(description="A script to process a dataset file.")

    parser.add_argument('-s', '--stack', type=str, required=True,
                        help='EvalStack.')
    parser.add_argument('-g', '--geo_layer', action='store_true',
                        help='Training geolayer expert.')
    parser.add_argument('-b', '--biclip', action='store_true',
                        help='Training BiCLIP model.')

    args = parser.parse_args()

    eval(args.stack, geolayer=args.geo_layer, biclip=args.biclip)