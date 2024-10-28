import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path

from ssl_libs.load_model import load_model, compute_features
from embed_utils         import compute_embeddings

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Traverse through the folders and collect image paths
        for folder, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other extensions as needed
                    full_path = os.path.join(folder, file)
                    self.image_paths.append(full_path)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Extract the folder and filename as requested
        folder_and_file_name = os.path.relpath(img_path, self.root_dir)
        return image, folder_and_file_name

def custom_collate_fn(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)

DATA_DIR   = '/home/bethge/cyildiz40/data/VidOR/imgs/'
MODEL_NAME = 'MAE'
RESULT_FOLDER = "object_embeddings"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model  = load_model(MODEL_NAME)

Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

# Example usage
# transform = transforms.Compose([transforms.ToTensor()])
transform  = T.Compose([T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset    = CustomImageDataset(root_dir=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

# Loop through the dataloader
EMBEDDINGS, FNAMES = [],[]
for i,(images,paths) in enumerate(dataloader):
    print(paths)  # Prints the list of (folder_and_file_name) strings
    embeddings = compute_embeddings(images, model, normalize=True)
    EMBEDDINGS.append(embeddings)
    FNAMES += paths
    if i%50==0:
        print(f'Saving at iter {i}')
        torch.save([torch.cat(EMBEDDINGS), FNAMES], f'{RESULT_FOLDER}/{MODEL_NAME}_embeddings.pt')