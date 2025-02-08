import os
import pandas as pd
from heatmap_conv.preprocess_dense import dense_optical_flow
from torch.utils.data.dataset import Dataset
import torch

class HeatmapDataset(Dataset):
    def __init__(self, video_dir, annotations_dir, config):
        self.annotations = pd.read_csv(annotations_dir)
        self.video_dir = video_dir
        self.image_width = config["image_width"]
        self.image_height = config["image_height"]
        self.magnitude_thresh = config["magnitude_thresh"]
        self.farneback_params = config["farneback_params"]
        self.farneback_params_str = "_".join([f"{value}" for _, value in self.farneback_params.items()])
        self.label_mapping = {label: idx for idx, label in enumerate(self.annotations.iloc[:, 1].unique())}
        self.cache_dir = os.path.join(self.video_dir, "cache_dir", "heatmap", self.farneback_params_str)
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):

        label_text = self.annotations.iloc[idx, 1]
        label = self.label_mapping[label_text]
        id = str(self.annotations.iloc[idx, 0])

        #check for cached file, compute heatmap if it doesnt exist
        cached_file_path = os.path.join(self.cache_dir, f"{id}.pt")
        folder_path = os.path.join(self.video_dir, id)

        if not os.path.exists(cached_file_path):

            heatmap = dense_optical_flow(
                folder_path=folder_path,
                img_height=self.image_height,
                img_width=self.image_width,
                farneback_params=self.farneback_params,
                magnitude_thresh=self.magnitude_thresh
            )

            os.makedirs(os.path.dirname(cached_file_path), exist_ok=True)
            heatmap = torch.tensor(heatmap)
            torch.save(heatmap, cached_file_path)

        else:
            heatmap = torch.load(cached_file_path, weights_only=False)
            
        if not isinstance(heatmap, torch.Tensor):
            heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return heatmap, label
