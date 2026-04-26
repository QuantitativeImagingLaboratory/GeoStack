from utils import seed_everything
from utils import logger
from utils import get_checkpoint_path
import argparse
from GeoStack.GeoStack import GeoStackCLIP
from GeoStack.GeoLayer import GeoLayer
import torch
from cil_dataloader import get_cifar100_cil_loaders
import clip
from tqdm import tqdm
from utils import write_results

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_layer(dataset, vit_model_name="ViT-B/16", device=None, geo_layer=False, biclip=False, task=None, total_tasks=None):
    checkpoint_path = get_checkpoint_path(dataset, scenario="cil", geolayer=geo_layer, biclip=biclip, task=task, total_tasks=total_tasks)
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


def evaluation(geolayer, biclip, num_tasks=4, forgetting=False):
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

    setting = "Learning" if not forgetting else "Forgetting"

    clip_model, preprocess = clip.load(backbone, device=device, jit=False)
    clip_model.float()
    clip_model.eval()
    dataset_name = "cifar100"
    _, test_loaders, prompt, classes = get_cifar100_cil_loaders(preprocess, num_tasks=num_tasks, num_shots=16)

    templates = [prompt % c for c in classes]
    text_tokens = clip.tokenize(templates).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)


    results = []
    for ind in range(len(test_loaders)):
        print(f"Evaluating on {ind+1}/{len(test_loaders)}")

        if zero_shot:
            logger.info(f"Zero-shot evaluation")
            model = GeoStackCLIP().to(device)
            model.float()

            results_file = f"{setting}_cil_{num_tasks}_zeroshot.csv"
        elif geolayer:
            print("Evaluating based on GeoLayers")
            all_geo_layers = []
            for k in range(ind + 1):
                all_geo_layers.append(get_layer(dataset_name, geo_layer=geolayer, biclip=biclip, task=k, total_tasks=num_tasks))
                model = GeoStackCLIP(clip_model=backbone, geo_layers=all_geo_layers).to(device)
            results_file = f"{setting}_cil_{num_tasks}_geostack.csv"
        elif biclip:
            print("Evaluating based on BiCLIP")
            all_geo_layers = []
            for k in range(ind + 1):
                all_geo_layers.append(get_layer(dataset_name, geo_layer=geolayer, biclip=biclip, task=k, total_tasks=num_tasks))
                model = GeoStackCLIP(clip_model=backbone, geo_layers=all_geo_layers).to(device)
            results_file = f"{setting}_cil_{num_tasks}_biclip.csv"

        evaluate_task = ind if not forgetting else 0 # Forgetting evaluate on task 0
        test_loader = test_loaders[evaluate_task]

        eval_model = model
        eval_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():

            try:
                for images, labels in tqdm(test_loader, desc="Evaluating"):
                    images = images.to(device).float()
                    labels = labels.to(device)

                    I_f = eval_model(images)
                    logits = eval_model.clip.logit_scale.exp() * I_f @ text_features.t()

                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)

        accuracy = 100 * correct / total
        results += [accuracy]
        logger.info(f"Train-on-task-{ind}: Evaluating-Task: {evaluate_task} Accuracy: {accuracy:.2f}%")

    write_data_list = []
    write_data = dict()

    for k in range(len(test_loaders)):
        temp_write_data = write_data.copy()
        eval_task = 0 if forgetting else k
        temp_write_data.update({"task": f"task-{eval_task}",
                                "accuracy": f"{results[k]:0.2f}"})


        logger.info(f"{setting}: Task-{eval_task} Accuracy: {results[k]:.2f}%")
        # print(f"Accuracy on {dataset_stack[k]}: {accuracies[k]}")
        write_data_list.append(temp_write_data)

    logger.info(f"Writing results to {results_file}")
    write_results(write_data_list, results_file)

if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser(description="A script to process a dataset file.")

    parser.add_argument('-n', '--num_tasks', type=int, default=4,
                        help='Number of tasks to train default(4).')
    parser.add_argument('-g', '--geo_layer', action='store_true',
                        help='Training geolayer expert.')
    parser.add_argument('-b', '--biclip', action='store_true',
                        help='Training BiCLIP model.')
    parser.add_argument('-f', '--forgetting', action='store_true',
                        help='Measure forgetting (default learning).')

    args = parser.parse_args()

    evaluation(geolayer=args.geo_layer, biclip=args.biclip, num_tasks=args.num_tasks, forgetting=args.forgetting)