
#%%
from pathlib import Path
import argparse

from linear_eval import linear_eval
from simclr import SimCLR

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file")
parser.add_argument("--data_dir", type=str, help="Path to the data directory", default="~/projects/ocl/data/PetFace/")
parser.add_argument("--log_dir", type=str, help="Path to the log directory", default="./logs/linear_eval/")
parser.add_argument("--devices", type=int, help="Number of devices to use", default=2)
parser.add_argument("--num_workers", type=int, help="Number of workers to use", default=16)
args = parser.parse_args()

# checkpoint = "./logs/simclr/5wci6ivs/checkpoints/epoch=24-step=15475.ckpt"
chkpt_path = Path(args.checkpoint).resolve()
model = SimCLR.load_from_checkpoint(chkpt_path)

#%%
NUM_CLASSES = 13
args.data_dir = Path(args.data_dir).expanduser().resolve()
args.log_dir = Path(args.log_dir).expanduser().resolve()
batch_size_per_device = model.batch_size_per_device

linear_eval(
    model=model.eval(),
    num_classes=NUM_CLASSES,
    train_dir=args.data_dir,
    val_dir=args.data_dir,
    log_dir=args.log_dir,
    batch_size_per_device=batch_size_per_device,
    num_workers=args.num_workers,
    accelerator="gpu",
    devices=args.devices,
)


# %%
