# GeoStack: A Framework for Abelian Knowledge Composition in VLMs

Official implementation of **GeoStack**, a modular framework for aggregating domain-specific expertise into Vision-Language Models (VLMs) with zero additional inference complexity.

GeoStack introduces **GeoLayers**—bilinear adapters that utilize geometric manifold constraints to enable **associative knowledge composition**. Multiple experts can be "folded" into a single weight matrix, ensuring that inference time remains constant ($O(1)$) regardless of the number of integrated tasks.

---

## 🚀 Key Features
- **Stackable Expertise:** Integrate $N$ domain experts (e.g., textures, satellite imagery, medical scans) into a single CLIP backbone.
- **Zero Overhead:** Expert composition is performed via matrix multiplication; the final model is as fast as the original CLIP.
- **Abelian Composition:** Knowledge integration is largely invariant to the order of tasks.
- **Theoretical Grounding:** Enforces upper-triangularity and near-isometry via Convex Orthogonality Alignment (COA).

---

## 🛠 Installation

```bash
git clone [https://https://github.com/QuantitativeImagingLaboratory/GeoStack](https://https://github.com/QuantitativeImagingLaboratory/GeoStack)
cd GeoStack
pip install -r requirements.txt
```

## 📂 Project Structure
```
├── configs/
│   ├── imagnet.yml    # Imagenet training config
│   ├── ...            # other dataset configs
├── GeoStack/
│   ├── GeoLayer.py    # GeoLayer model
│   └── GeoStack.py    # GeoStack model to compose GeoLayers
├── losses.py          # Contians loss functions
├── mda_train.py       # Training for Multi-Domain Adaptation
├── mda_eval.py        # Evaluation of stacked experts
├── cil_train.py       # Training for Class-Incremental Learning on CIFAR-100 
├── cil_eval.py        # Long-term stability & forgetting benchmarks
└── utils.py           # Metrics (Orthogonality Error) and checkpointing
```

## 🏋️ Training an Expert
To train a single GeoLayer expert on a specific domain (e.g., DTD or imagenet):

#### Multi-Domain Adaptation (MDA)
```
python mda_train.py --dataset dtd --geo_layer

--geo_layer: Enables geometric constraints (Upper-triangularity + COA loss).
--biclip: Trains a standard bilinear adapter baseline without stacking constraints.
```

#### Class-Incremental Learning (CIL)
```
python cil_train.py --geo_layer --num_tasks 4

--geo_layer: Enables geometric constraints (Upper-triangularity + COA loss).
--biclip: Trains a standard bilinear adapter baseline without stacking constraints.
--num_tasks: Specify number of tasks to train on
```

## 📚 Stacking and Evaluation
#### Multi-Domain Adaptation (MDA)
Evaluate how multiple experts perform when folded together. Use the --stack argument to define the sequence of experts:

```
python mda_eval.py --stack 'i->d' --geo_layer # Stack imagenet and dtd geolayers

--stack: specify stack seperated by arror '->'
--geo_layer: Uses GeoLayers for evaluation
--biclip: Uses BiCLIP layers for evaluation
```
#### Class-Incremental Learning (CIL)
Evaluate the resilience to catastrophic forgetting across sequential tasks:

```
# Evaluate accuracy each cumulative task after training 10 tasks
python cil_eval.py --num_tasks 10 --geo_layer
 
# Evaluate accuracy on task 0 after training 10 tasks
python cil_eval.py --num_tasks 10 --geo_layer --forgetting

--geo_layer: Uses GeoLayers for evaluation
--biclip: Uses BiCLIP layers for evaluation
--forgetting: Set true to evaluate only on the first task
```

## 📐 Core Concept: 
Folding ExpertsGeoStack relies on the associative property of matrix multiplication. If $W_1$ and $W_2$ are GeoLayer weights for Task 1 and Task 2, the combined expert $W_{stack}$ is:$$W_{stack} = W_1 \cdot W_2$$In code, this is handled by the GeoStackCLIP wrapper:Pythonfrom GeoStack.GeoStack import GeoStackCLIP
```
from GeoStack.GeoLayer import GeoLayer


expert1 = GeoLayer(embed_dim=512) # Load Expert 1
checkpoint = torch.load(checkpoint_path, map_location=device)
expert1.load_state_dict(checkpoint['model_state_dict'])

expert2 = GeoLayer(embed_dim=512) # Load Expert 1
checkpoint = torch.load(checkpoint_path, map_location=device)
expert2.load_state_dict(checkpoint['model_state_dict'])

model = GeoStackCLIP(clip_model="ViT-B/16", geo_layers=[expert1, expert2]) # Fold them into a single CLIP model

#### Inference 
logits = model(images)
```

# 📝 Citation
GeoStack
```
@inproceedings{anonymous2024geostack,
  title={GeoStack: A Framework for Abelian Knowledge Composition in VLMs},
  author={Anonymous},
  booktitle={Preprint},
  year={2024}
}
```
