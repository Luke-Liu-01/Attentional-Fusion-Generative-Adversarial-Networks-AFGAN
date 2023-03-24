import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from gensim.models.keyedvectors import KeyedVectors

default_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),  # [0, 255] -> [0, 1]
    ])

def get_img_path(root_path):
    img_paths = []
    for folder in os.listdir(root_path):
        file_path = os.path.join(root_path, folder)
        for file in os.listdir(file_path):
            img_paths.append(os.path.join(file_path, file).replace('\\', '/'))
    return img_paths


def get_captions(img_paths, metadata_path):
    with open(metadata_path, 'r') as f:
        idx2caption = json.loads(f.read())
    captions = []
    stop_words = ['a', 'an','and', 'as', 'with', 'of', 'if', 'for','its','be', 'to', 'may', 'from', 'or', 'the']
    for img_path in img_paths:  # e.g. '../data/emoji/emoji_faces/Apple/Apple_10.png'
        img_name = img_path.split('/')[-1]
        idx = img_name.split('.')[-2].split('_')[-1]
        caption = idx2caption[idx]
        caption = [word for word in caption.split() if word.lower() not in stop_words]  # filter stop words
        caption = ' '.join(caption)
        captions.append(caption)
    return captions

def embedding_vec(captions, wordvector_path):
    word2vec = KeyedVectors.load_word2vec_format(wordvector_path, binary=True)
    caption_vectors = []
    for caption in captions:
        wv = 0.
        for token in caption.split():
            try:
                wv += word2vec[token]
            except:
                continue
        caption_vectors.append(torch.tensor(wv))  # ndarray to tensor
    return caption_vectors

def embedding_mat(captions, wordvector_path, max_length):
    word2vec = KeyedVectors.load_word2vec_format(wordvector_path, binary=True)
    embedding_dim = word2vec['token'].shape[0]
    caption_mats = []
    for caption in captions:
        caption_mat = torch.zeros((max_length, embedding_dim))  # initialize
        for i, token in enumerate(caption.split()):
            try:
                caption_mat[i] = word2vec[token]
            except:
                continue
        caption_mats.append(caption_mat)
    return caption_mats


class EmojiSet(Dataset):

    def __init__(self, data_path, metadata_path, wordvector_path, transforms=None):

        self.img_paths = get_img_path(data_path)
        self.captions = get_captions(self.img_paths, metadata_path)
        self.caption_vectors = embedding_vec(self.captions, wordvector_path)
        self.transforms = transforms if transforms is not None else default_transforms
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transforms(img)
        caption = self.captions[index]
        caption_vec = self.caption_vectors[index]
        return img, caption, caption_vec

    def __len__(self):
        return len(self.img_paths)


# using the sentence matrix
class EmojiSet_TIMM(Dataset):

    def __init__(self, data_path, metadata_path, wordvector_path, max_length, transforms=None):

        self.img_paths = get_img_path(data_path)
        self.captions = get_captions(self.img_paths, metadata_path)
        self.caption_matrices = embedding_mat(self.captions, wordvector_path, max_length)
        self.transforms = transforms if transforms is not None else default_transforms
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transforms(img)
        caption = self.captions[index]
        caption_matrix = self.caption_matrices[index]
        return img, caption, caption_matrix

    def __len__(self):
        return len(self.img_paths)