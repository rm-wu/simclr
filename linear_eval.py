import argparse
from datetime import datetime
import torch
from pathlib import Path

from eval import seed_everything
from src.models import get_ibot_model_by_name
from src.ls_eval import ls_finetune


def main(args):
    print(f"Linear Segmentation arguments: {args}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## Load pretrained model
    if args.model.startswith("dinov2"):
        # TODO: this part needs to be checked to see if it can handle also the registers
        model = torch.hub.load("facebookresearch/dinov2", args.model)
    elif args.model.startswith("dino"):
        model = torch.hub.load("facebookresearch/dino:main", args.model)
    elif args.model.startswith("ibot"):
        model = get_ibot_model_by_name(args.model)
    else:
        raise ValueError(f'Model "{args.model}" not recognized')
    model = model.to(device)

    if args.model.startswith("dinov2"):

        def token_features(model, imgs):
            return model.get_intermediate_layers(imgs)[0], None

    else:

        def token_features(model, imgs):
            return model.get_intermediate_layers(imgs)[0][:, 1:], None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = (
        Path(args.out_dir)
        / args.dataset_name
        / f"{args.model}_i{args.input_size}_p{args.patch_size}_e{args.embeddings_size}_b{args.batch_size}_s{args.seed}"
        / timestamp
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ls_finetune(
        backbone=model,
        patch_size=args.patch_size,
        head_type="linear",
        max_epochs=args.max_epochs,
        lr=args.lr,
        decay_rate=0.1,
        drop_at=args.drop_at,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        feat_extr_fn=token_features,
        input_size=args.input_size,
        train_mask_size=100,
        val_mask_size=100,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear Segmentation Evaluation")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--embeddings-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=14)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument(
        "--drop_at", type=int, default=20
    )  # TODO: check what is this about

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="voc",
        choices=["voc", "ade20k", "coco-thing", "coco-stuff"],
    )
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--out-dir", type=str, default="outputs/linear_eval/")
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Whether to save the features and labels to the output directory",
    )

    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)
