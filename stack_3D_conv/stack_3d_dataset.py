import os
import pandas as pd
from preprocess_3d import get_last_img, get_random_img, compute_vectors_from_folder
from preprocess_3d import apply_rotation_aug, apply_scaling_aug
from torch.utils.data.dataset import Dataset
import random
import torch


class Stacked3DDataset(Dataset):
    def __init__(self, video_dir, annotations_dir, spatial_stream, config, augment):
        self.annotations = pd.read_csv(annotations_dir)
        self.spatial_stream = spatial_stream
        self.video_dir = video_dir
        self.image_width = config["image_width"]
        self.image_height = config["image_height"]
        self.magnitude_thresh = config["magnitude_thresh"]
        self.farneback_params = config["farneback_params"]
        self.max_flows = config["max_flows"]
        self.farneback_params_str = "_".join([f"{value}" for _, value in self.farneback_params.items()])
        self.label_mapping = {label: idx for idx, label in enumerate(self.annotations.iloc[:, 1].unique())}
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        label_text = self.annotations.iloc[idx, 1]
        label = self.label_mapping[label_text]
        id = str(self.annotations.iloc[idx, 0])

        folder_path = os.path.join(self.video_dir, id)

        if self.spatial_stream:
            random_img = get_random_img(folder_path = folder_path, img_height = self.image_height, img_width = self.image_width, augment = self.augment)
            # random_img = get_last_img(folder_path = folder_path)
            random_img = torch.tensor(random_img)
            return random_img, label

        else:
            stacked_vectors = compute_vectors_from_folder(
                folder_path=folder_path,
                img_height=self.image_height,
                img_width=self.image_width,
                max_flows=self.max_flows,
                farneback_params=self.farneback_params,
                magnitude_thresh=self.magnitude_thresh,
                augment = self.augment
            )

            stacked_vectors = torch.tensor(stacked_vectors, dtype=torch.float32)
            return stacked_vectors, label
