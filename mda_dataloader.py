from torchvision.datasets import ImageFolder
from torchvision.datasets import OxfordIIITPet, FGVCAircraft, Flowers102, ImageNet
from torchvision.datasets import (
    StanfordCars, Food101, DTD,
    EuroSAT, Caltech101,
)
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from utils import ApplyTransform
from utils import *
from pathlib import Path
from utils import logger


def get_flower_names(i):
    flower_names = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
        "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
        "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
        "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
        "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
        "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
        "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
        "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
        "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
        "mexican petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff",
        "gaura", "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
        "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus",
        "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose",
        "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
        "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
        "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
        "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
        "trumpet creeper", "blackberry lily"
    ]

    return flower_names[i]

def get_eurosat_classes(input_list):
    class_map = {
        "annualcrop": "annual crop land",
        "forest": "forest",
        "herbaceousvegetation": "herbaceous vegetation land",
        "highway": "highway or road",
        "industrial": "industrial buildings",
        "pasture": "pasture land",
        "permanentcrop": "permanent crop land",
        "residential": "residential buildings",
        "river": "river",
        "sealake": "sea or lake"
    }

    return [class_map[k.lower()] for k in input_list]



def get_dataset(dataset_name, preprocess, batch_size=64, include_labels=False, num_shots=-1):

    root = os.path.expanduser("~/.cache")

    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    if dataset_name == "Aircraft".lower():
        train_ds = FGVCAircraft(root, split="trainval", download=True, transform=train_preprocess)
        test_ds = FGVCAircraft(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        labels = test_ds._labels

        prompt = "a photo of an %s, a type of aircraft."
    elif dataset_name == "OxfordPet".lower():
        train_ds = OxfordIIITPet(root, split="trainval", download=True, transform=train_preprocess)
        test_ds = OxfordIIITPet(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        labels = test_ds._labels
        prompt = "a photo of a %s, a type of pet."
    elif dataset_name == "Flowers102".lower():
        train_ds_ = Flowers102(root, split="train", download=True, transform=train_preprocess)
        val_ds = Flowers102(root, split="val", download=True, transform=train_preprocess)

        train_ds = torch.utils.data.ConcatDataset([train_ds_, val_ds])
        test_ds = Flowers102(root, split="test", download=True, transform=preprocess)

        classes = [f"{get_flower_names(i)}" for i in range(102)]
        labels = test_ds._labels
        prompt = "a photo of a %s, a type of flower."
    elif dataset_name == "caltech101":
        full_ds = Caltech101(root, download=True, transform=None)

        def get_few_shot_split(full_ds, n_shots=16):
            indices = np.arange(len(full_ds))
            labels = [full_ds[i][1] for i in indices]

            train_indices = []
            test_indices = []

            for label in range(len(full_ds.categories)):
                label_indices = [i for i, l in enumerate(labels) if l == label]
                # Sample exactly 16 for training, everything else for testing
                np.random.shuffle(label_indices)
                train_indices.extend(label_indices[:n_shots])
                n_shot_for_test_set = 16
                test_indices.extend(label_indices[n_shot_for_test_set:n_shot_for_test_set + 19])

            return Subset(full_ds, train_indices), Subset(full_ds, test_indices)

        train_ds, test_ds = get_few_shot_split(full_ds, n_shots=num_shots)

        train_ds = ApplyTransform(train_ds, transform=train_preprocess)
        test_ds = ApplyTransform(test_ds, transform=preprocess)

        classes = full_ds.categories

        prompt = "a centered photo of a %s."

    elif dataset_name == "dtd":
        train_ds = DTD(root, split="train", download=True, transform=train_preprocess)
        test_ds = DTD(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        prompt = "a photo of a %s texture."

    elif dataset_name == "eurosat":
        full_ds = EuroSAT(root, download=True, transform=None)

        train_size = int(0.8 * len(full_ds))
        test_size = len(full_ds) - train_size
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

        train_ds = ApplyTransform(train_ds, transform=train_preprocess)
        test_ds = ApplyTransform(test_ds, transform=preprocess)
        classes = full_ds.classes
        classes = get_eurosat_classes(classes)
        prompt = "a centered satellite photo of %s."

    elif dataset_name == "food101":
        train_ds = Food101(root, split="train", download=True, transform=train_preprocess)
        test_ds = Food101(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        prompt = "a photo of %s, a type of food."

    elif dataset_name == "ImageNet".lower():
        if os.getenv("SABINE", False):
            train_root = "imagenet/partial16"
        else:
            train_root = None

        assert train_root is not None, f"{dataset_name} path is not defined"

        train_ds = ImageFolder(
            root=train_root,
            transform=train_preprocess  # Use the CLIP preprocess function
        )

        test_ds = ImageNet(root=root, split="val", transform=preprocess)
        all_classes = get_imagenet_classes()
        classes = [label.replace('\n', '').strip() for label in all_classes]

        prompt = "a photo of a %s"

    elif dataset_name == "stanfordcars":

        train_ds = StanfordCars(root, split="train", download=False, transform=train_preprocess)
        test_ds = StanfordCars(root, split="test", download=False, transform=preprocess)
        all_classes = train_ds.classes

        def process_stanford_cars_classes(classname):
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            return ' '.join(names)

        classes = [process_stanford_cars_classes(c) for c in all_classes]

        prompt = "a photo of %s, a type of car."

    elif dataset_name == "sun397":
        from datasets import load_from_disk

        # Point to the folder contianing the dataset
        small_path = None
        assert small_path is not None, f"{dataset_name} Dataset is not defined. Requires manual download."

        small_bundle = load_from_disk(small_path)

        def process_class_name(raw_name):
            names = raw_name.split('/')[2:]
            names = names[::-1]
            classname = ' '.join(names)
            return classname

        classes_all = small_bundle['train'].features["label"].names
        classes = [process_class_name(c) for c in classes_all]

        class HFDatasetWrapper(Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.hf_dataset = hf_dataset
                self.transform = transform
                self._labels = self.hf_dataset['label']

            def __getitem__(self, index):
                item = self.hf_dataset[index]
                x, y = item['image'], item['label']
                if hasattr(x, 'convert'): x = x.convert("RGB")
                if self.transform: x = self.transform(x)
                return x, y

            def __len__(self):
                return len(self.hf_dataset)

        train_ds = HFDatasetWrapper(small_bundle['train'], transform=train_preprocess)
        test_ds = HFDatasetWrapper(small_bundle['test'], transform=preprocess)

        prompt = "a centered photo of %s."

    elif dataset_name == "ucf101":
        d_path = None
        s_path = None

        assert d_path is not None, f"{dataset_name} path is not defined"
        assert s_path is not None, f"{dataset_name} path is not defined"

        data_path = Path(d_path)
        split_dir = Path(s_path)

        train_list_file = split_dir / "trainlist01.txt"
        test_list_file = split_dir / "testlist01.txt"

        def load_split(list_file, is_test=False):
            samples = []
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    video_rel_path = parts[0]

                    img_rel_path = video_rel_path.replace('.avi', '.jpg')
                    img_full_path = data_path / img_rel_path

                    class_name = video_rel_path.split('/')[0]

                    samples.append((str(img_full_path), class_name))
            return samples

        train_samples = load_split(train_list_file)
        test_samples = load_split(test_list_file, is_test=True)

        all_classes = sorted(list(set([s[1] for s in train_samples + test_samples])))
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

        class FileListDataset(Dataset):
            def __init__(self, samples, class_to_idx, transform=None):
                self.samples = samples
                self.class_to_idx = class_to_idx
                self.transform = transform

            def __len__(self): return len(self.samples)

            def __getitem__(self, i):
                path, cls_name = self.samples[i]
                from PIL import Image
                img = Image.open(path).convert("RGB")
                if self.transform: img = self.transform(img)
                return img, self.class_to_idx[cls_name]

        train_ds = FileListDataset(train_samples, class_to_idx, transform=train_preprocess)
        test_ds = FileListDataset(test_samples, class_to_idx, transform=preprocess)

        classes = all_classes
        import re
        classes = [re.sub(r'(?<!^)(?=[A-Z])', ' ', c) for c in classes]
        prompt = "a photo of a person %s."

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    logger.info(f"{dataset_name.upper()}: Processing for {num_shots}-shot classification.")
    if num_shots > 0:
        if hasattr(train_ds, 'targets'):
            labels = np.array(train_ds.targets)
        elif hasattr(train_ds, '_labels'):
            labels = np.array(train_ds._labels)
        else:
            first_item = train_ds[0]
            if isinstance(first_item, dict):
                labels = np.array([train_ds[i]['label'] for i in range(len(train_ds))])
            else:

                labels = np.array([train_ds[i][1] for i in range(len(train_ds))])

        few_shot_indices = []
        unique_classes = np.unique(labels)

        for c in unique_classes:
            indices = np.where(labels == c)[0]

            replace = False
            if dataset_name == 'flowers102':
                replace = True
            sampled = np.random.choice(indices, num_shots, replace=replace)
            few_shot_indices.extend(sampled)

        train_ds = Subset(train_ds, few_shot_indices)
        logger.info(f"{dataset_name.upper()}: Created {num_shots}-shot dataset with {len(train_ds)} total images.")

    clean_classes = [c.replace("_", " ").replace("-", " ").lower() for c in classes]

    if dataset_name == 'imagenet':
        batch_size = 256

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if include_labels:
        return train_loader, test_loader, prompt, clean_classes, labels
    else:
        return train_loader, test_loader, prompt, clean_classes


def get_imagenet_classes():
    imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                        "box turtle", "banded gecko", "green iguana", "Carolina anole",
                        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                        "American alligator", "triceratops", "worm snake", "ring-necked snake",
                        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
                        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
                        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                        "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                        "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
                        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
                        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

    return imagenet_classes

