import torch
from torch.utils.data import DataLoader
import os
from example import VideoDataset, video_names, transform

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
model.eval()  # Set model to evaluation mode

# Create dataset and dataloader
video_dataset = VideoDataset(
    video_names,
    num_videos=5, 
    transform=transform
)
data_loader = DataLoader(video_dataset, batch_size=1, shuffle=True)

def extract_features(frame):
    """Extract features from a single frame using DINOv2"""
    with torch.no_grad():
        frame = frame.to(device)
        output = model(frame, is_training=False)
        return output["x_prenorm"]  # Shape: [B, num_patches, embedding_dim]

# Process a batch of frames
for batch_idx, (frame1, frame2) in enumerate(data_loader):
    # Each frame is a tuple of (image, segmentation_map)
    frame1_img, frame1_seg = frame1
    frame2_img, frame2_seg = frame2
    
    # Extract features
    features1 = extract_features(frame1_img)
    features2 = extract_features(frame2_img)
    
    print(f"Batch {batch_idx}")
    print(f"Features shape: {features1.shape}")
    
    # Break after first batch for testing
    break


