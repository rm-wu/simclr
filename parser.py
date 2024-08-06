import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/projects/ocl/data/PetFace/",
        help="Data directory",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logger"
    )
    parser.add_argument(
        "--resume_chkpt", type=str, default=None, help="Path to checkpoint to resume"
    )   
    
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--train_batchsize", type=int, default=512, help="Train batch size"
    )
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer",
        choices=["SGD", "LARS"],
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.06, help="Learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    parser.add_argument(
        "--natural_augmentation",
        action="store_true",
        help="Use different views from the PetFace dataset as augmentations",
    )

    ## SimCLR argumentation
    parser.add_argument("--input_size", type=int, default=224, help="Image resize")
    parser.add_argument(
        "--cj_prob", type=float, default=0.8, help="Color jitter probability"
    )
    parser.add_argument(
        "--cj_strength", type=float, default=1.0, help="Color jitter strength"
    )
    parser.add_argument(
        "--cj_bright", type=float, default=0.8, help="Color jitter brightness"
    )
    parser.add_argument(
        "--cj_contrast", type=float, default=0.8, help="Color jitter contrast"
    )
    parser.add_argument(
        "--cj_sat", type=float, default=0.8, help="Color jitter saturation"
    )
    parser.add_argument("--cj_hue", type=float, default=0.2, help="Color jitter hue")
    parser.add_argument(
        "--min_scale", type=float, default=0.08, help="Crop minimum scale"
    )
    parser.add_argument(
        "--random_gray_scale", type=float, default=0.2, help="Random gray scale"
    )
    parser.add_argument(
        "--gaussian_blur", type=float, default=0.5, help="Gaussian blur"
    )

    parser.add_argument(
        "--accelerator", default="gpu", choices=["gpu", "cpu"], help="Use GPU or CPU"
    )
    parser.add_argument("--devices", default=2, type=int, help="Number of devices")

    args = parser.parse_args()
    return args
