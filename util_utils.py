import torch
import os

def print_gpu_memory():
    try:
        # print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        # print(f"Memory Reserved (Cached): {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")
        print(f"Free GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2 - torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        # print("--------------------")
    except:
        pass

def get_root_folder():
    if 'cagatay' in os.getcwd():
        return 'data/VidOR'
    else:
        if os.path.exists('/weka'):
            return '/home/bethge/cyildiz40/data/VidOR'
        else:
            return '/mnt/lustre/work/bethge/cyildiz40/projects/videossl/VidOR'

def get_device(model):
    return list(model.parameters())[0].device

def apply_transform(frames, transform):
    ''' frames could be channel first or last'''
    # assert frames.shape[-3]==3, 'channel dimension wrong'
    if frames.shape[-3]!=3:
        if frames.ndim==3:
            frames = frames.permute(2,0,1) # 3, width, height
            transformed_frames = transform(frames).permute(1,2,0) # 224, 224, 3
        elif frames.ndim==4:
            frames = frames.permute(0,3,1,2) # N, 3, width, height
            transformed_frames = transform(frames).permute(0,2,3,1) # N, 224, 224, 3
        elif frames.ndim==5:
            frames = frames.permute(0,1,4,2,3) # N, T, 3, width, height
            transformed_frames = transform(frames).permute(0,1,3,4,2) # N, T, 224, 224, 3
    else:
        transformed_frames = transform(frames)
    return transformed_frames # [N,T,224,224,3] or [N,224,224,3]  or [224,224,3]
