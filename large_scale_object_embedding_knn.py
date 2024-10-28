# list all the folders in a direc§:tory
import os, cv2, numpy as np

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def same_class(fname1, fname2):
	video1, obj1 = fname1.split('/')[0], fname1.split('/')[1].split('_')[0]
	video2, obj2 = fname2.split('/')[0], fname2.split('/')[1].split('_')[0]
	return video1==video2 and obj1==obj2

for MODEL_NAME in ['CLIP', 'DINOv2-reg', 'MAE']:
	embeddings, fnames = torch.load(f'object_embeddings/{MODEL_NAME}_embeddings.pt')

	N,q = embeddings.shape
	S = 1000
	misclassified_anc, misclassified_targets = [],[]

	for s in range(0,N,S):
		similarity = torch.zeros(N,S).to(device)
		print(s//S, '/', N//S, 'class_error:', ((len(misclassified_anc)+1)/(s+1)))
		slice_ = min(N-s,S)
		inputs = embeddings[s:s+slice_]
		for i in range(0,N,S):
			slice__ = min(N-i,S)
			similarity[i:i+slice_] = (embeddings[i:i+slice__].unsqueeze(0) * inputs.unsqueeze(1)).sum(-1).T
		similarity[s+torch.arange(S),torch.arange(S)] = -1
		_,nearest_neighbors = similarity.topk(5,dim=0)
		for j in range(similarity.shape[1]):
			if not same_class(fnames[nearest_neighbors[0,j]], fnames[s+j]):
				misclassified_anc.append(fnames[s+j])
				misclassified_targets.append([fnames[n] for n in nearest_neighbors[:,j]] )

	torch.save([misclassified_anc,misclassified_targets],f'object_embeddings/{MODEL_NAME}_misclassified_examples.pt')
