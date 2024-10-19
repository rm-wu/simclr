import os, cv2, numpy as np

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

def get_root_folder():
    if 'cagatay' in os.getcwd():
        return '/Users/cagatay/Downloads/VidOR'
    else:
        if os.path.exists('/weka'):
            return '/home/bethge/cyildiz40/data/VidOR'
        else:
            return '/mnt/lustre/work/bethge/cyildiz40/projects/videossl/VidOR'

ROOT = get_root_folder()
video_names = os.listdir(os.path.join(ROOT, 'videos'))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print(ROOT, device)
