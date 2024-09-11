
from PIL import Image
import torch
import torch.nn as nn
import os,numpy as np
from torchvision import transforms as T
from methods.simclr import SimCLR
from lightly.transforms.utils import IMAGENET_NORMALIZE
from parser import parse_arguments

args = parse_arguments()
print(f"Command line arguments {args}")

MAIN_FOLDER = args.data_dir
all_classes = os.listdir(os.path.join(args.data_dir,'split'))

transform = T.Compose(
    [   
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ]
)

# read 256 images from each class
IMG_PER_CLASS = 256
IMAGES,LABELS = [],[]
for class_ in all_classes:
    images,labels,unique_animal_ids = [],[],[]
    class_folder = os.path.join(args.data_dir, 'split', class_, 'val.txt')
    with open(class_folder, 'r') as f:
        lines = f.readlines()[:IMG_PER_CLASS]
        for line in lines:
            img_path = line.strip()
            # example path: pig/001170/01.png. get the middle part, which is the animal name
            animal_id = img_path.split('/')[1]
            if animal_id not in unique_animal_ids:
                unique_animal_ids.append(animal_id)
            image = Image.open(os.path.join(args.data_dir, 'images', img_path))
            images.append(transform(image))
            labels.append(len(unique_animal_ids)-1)
    IMAGES.append(torch.stack(images))  # [256,3,224,224]
    LABELS.append(torch.tensor(labels)) # [256]
    print(class_, 'completed')

try:
    model = SimCLR.load_from_checkpoint(args.ckpt_path)
except:
    # model = SimCLR.load_from_checkpoint('/mnt/qb/work/bethge/cyildiz40/simclr/logs/lightning/petface/epoch=9-step=48000.ckpt')
    model = SimCLR.load_from_checkpoint('/mnt/qb/work/bethge/cyildiz40/simclr/logs/lightning/petface-nat/epoch=8-step=42000.ckpt')

def animal_identity_identify(num_neg):
    while True:
        # pick a random class
        c = np.random.randint(len(all_classes))
        # pick a random image from that class
        anchor = np.random.randint(IMAGES[c].shape[0])
        anchor_label = LABELS[c][anchor]
        # pick a positive image from the same class
        positives = torch.where(LABELS[c]==anchor_label)[0]
        positives = positives[positives!=anchor]
        if len(positives)>0:
            pos = positives[torch.randperm(len(positives))[0]].item()
            break
    # pick num_neg negative images from the same class
    neg_idx = torch.where(LABELS[c]!=LABELS[c][anchor])[0]
    neg_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]
    # concat all images
    all_images = torch.cat([IMAGES[c][anchor:anchor+1], IMAGES[c][pos:pos+1], IMAGES[c][neg_idx]])
    return all_images,c

def animal_type_identify(num_neg):
    # pick two random classes
    c,c_ = torch.randperm(len(all_classes))[:2]
    # pick two images from the first class
    anchor = np.random.randint(IMAGES[c].shape[0])
    pos = np.random.randint(IMAGES[c].shape[0])
    # pick num_neg negative images from the second class
    negatives = IMAGES[c_][torch.randperm(len(IMAGES[c_]))[:num_neg]]
    all_images = torch.cat([IMAGES[c][anchor:anchor+1], IMAGES[c][pos:pos+1], negatives]) # anchor, pos, negs
    return all_images,c

answers = [[]]*len(all_classes)
num_neg = 62
for i in range(500):
    # all_images,c = animal_identity_identify(num_neg)
    all_images,c = animal_type_identify(num_neg)
    with torch.no_grad():
        Z = model.backbone(all_images.to(model.device))[:,:,0,0]
        Z = model.projection_head(Z)
    Z = nn.functional.normalize(Z, dim=1)
    sims = (Z[0] * Z[1:]).sum(-1)
    acc  = (sims.argmax()==0).int().item()
    # print(c,acc)
    answers[c] = answers[c] + [acc]

print([float(np.mean(ans)) for ans in answers])
print(np.mean([a for ans in answers for a in and]))


# def plot_10_images(all_images,most_sims,c):
#     fig,axs = plt.subplots(1,10,figsize=(20,2))
#     axs[0].imshow(all_images[0].permute(1,2,0)*torch.tensor(IMAGENET_NORMALIZE["std"]).unsqueeze(0).unsqueeze(0) + torch.tensor(IMAGENET_NORMALIZE["mean"]).unsqueeze(0).unsqueeze(0))
#     for i in range(9):
#         axs[i+1].imshow(all_images[most_sims[i]].permute(1,2,0)*torch.tensor(IMAGENET_NORMALIZE["std"]).unsqueeze(0).unsqueeze(0) + torch.tensor(IMAGENET_NORMALIZE["mean"]).unsqueeze(0).unsqueeze(0))
#         axs[i+1].axis('off')
#     plt.tight_layout()
#     plt.savefig(f'same_images_{c}.png')
#     plt.close()


# simclr - with head - 0.33
# simclr - w/o projection head - 0.342
# ours   - with head - 0.514
# ours   - w/o projection head - 0.536
