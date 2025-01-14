import os
import torch
import random
import argparse
import numpy as np

from src.hbird_eval import hbird_evaluation
from src.ibot_vision_transformer import get_ibot_model_by_name


def main(args):
    print(f"the script arguments are {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model.startswith("dinov2"):
        model = torch.hub.load('facebookresearch/dinov2', args.model).to(device)
    elif args.model.startswith("dino"):
        model = torch.hub.load("facebookresearch/dino:main", args.model).to(device)
    elif args.model.startswith("ibot"):
        model = get_ibot_model_by_name(args.model).to(device)
    else:
        raise ValueError("Model not recognized")
        
    
    # dinov2 return a tuple of (features, logits)

    if args.model.startswith("dinov2"):
        def token_features(model, imgs):
            return model.get_intermediate_layers(imgs)[0], None
    else:
        # dino and ibot
        def token_features(model, imgs):
            # for dino and ibot, we unselect the first token which is the [CLS] token
            return model.get_intermediate_layers(imgs)[0][:, 1:], None
    

    
    dataset_name = "voc"
    aug_epoch = 1
    out_dir = os.path.join(args.out_dir, dataset_name, f"{args.model}_i{args.input_size}_p{args.patch_size}_e{args.embeddings_size}_m{args.memory_size}_a{aug_epoch}_b{args.batch_size}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    hbird_miou = hbird_evaluation(
        model.to(device),
        # Size of the embedding feature vectors of patches
        d_model=args.embeddings_size,
        patch_size=args.patch_size,
        batch_size = args.batch_size,
        input_size=args.input_size,
        # How many iterations of augmentations to use on top of the training dataset in order to generate the memory
        augmentation_epoch=aug_epoch,
        device=device,
        # Whether to return additional NNs details
        return_knn_details=False,
        # The number of neighbors to fetch per image patch
        n_neighbours=30,
        # Other parameters to be used for the k-NN operator
        nn_params=None,
        # Function that extracts features from a vision encoder on images
        ftr_extr_fn=token_features,
        # The name of the dataset to use, currently only Pascal VOC is included.
        dataset_name=dataset_name,
        # Path to the dataset to use for evaluation
        data_dir=args.data_dir,
        out_dir=out_dir,
        memory_size=args.memory_size if args.memory_size > -1 else None,
    )
    np.save(os.path.join(out_dir, "hbird_miou.npy"), np.array([hbird_miou]))
    print(f"Hummingbird Evaluation (mIoU): {hbird_miou}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")
    parser.add_argument("--patch-size", type=int, default=16, help="Size of the model patches")
    parser.add_argument("--memory-size", type=int, default=None, help="The size of the memory bank. Unbounded if not specified")
    parser.add_argument("--model", type=str, required=True, help="DINO model name")
    parser.add_argument("--embeddings-size", type=int, required=True, help="The size of the model embeddings")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="VOCSegmentation", help="Path to the VOC dataset")
    parser.add_argument("--out-dir", type=str, default="out-cribo", help="Path to the output directory")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
