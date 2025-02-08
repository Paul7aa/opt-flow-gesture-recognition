import pandas as pd
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

csv_files = ["data/jester-v1-train.csv",
            "data/jester-v1-validation.csv"] # test set isn't publicly available

labels = ["Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up",
           "Zooming In With Two Fingers" , "Zooming Out With Two Fingers"]

def create_subset():
    if not os.path.exists("subset/"):
        os.mkdir("subset/")

def extract_data():
    aggregated_df = pd.DataFrame(columns=["ID", "Label"])

    for csv_file in csv_files:
        dataframe = pd.read_csv(csv_file, sep=';', header=None, names=["ID", "Label"])
        for label in labels:
            label_df = dataframe.loc[dataframe["Label"] == label]
            aggregated_df = pd.concat([aggregated_df, label_df], ignore_index=True)
    
    #add extra "Doing something else label" with randomly sampled examples
    #to keep balanced dataset
    extra_size = aggregated_df.size // len(labels) #labels to add
    extra_df = dataframe.loc[~dataframe["Label"].isin(labels)].sample(n=extra_size, random_state=42)
    extra_df["Label"] = "Doing something else"
    
    aggregated_df = pd.concat([aggregated_df, extra_df], ignore_index=True)

    aggregated_df.to_csv("jester_subset/subset.csv", index=False)

def copy_data():
    df = pd.read_csv("subset/subset.csv")

    for i, row in df.iterrows():
        id = row["ID"]
        print(f"Copying folder with index {i}")
        source_folder = os.path.join("data", "20bn-jester-v1", str(id))
        destination = os.path.join("subset", "data", str(id))
        shutil.copytree(source_folder, destination)


def split_data():
    df = pd.read_csv("subset/subset.csv")

    # 80% training - 10% validation - 10% test 
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv("jester_subset/train.csv")
    val_df.to_csv("jester_subset/validation.csv")
    test_df.to_csv("jester_subset/test.csv")

def count_avg_frames():
    frames = []
    max_frames = 0
    min_frames = float("inf")
    df = pd.read_csv("rec_dataset2/data.csv")

    for _, video in df.iterrows():
        print(len(df))
        nr_frames = len(os.listdir(f"rec_dataset2/{video["ID"]}"))
        frames.append(nr_frames)
        if nr_frames > max_frames:
            max_frames = nr_frames
        if nr_frames < min_frames:
            min_frames = nr_frames

    print("AVERAGE FRAMES: ", np.average(np.array(frames)))
    print("MINIMUM FRAMES: ", min_frames)
    print("MAXIMUM FRAMES: ", max_frames )

        
def combine_datasets_ordered_by_id(dataset1_path, dataset2_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Get all existing IDs in dataset1
    dataset1_ids = set([d for d in os.listdir(dataset1_path) if os.path.isdir(os.path.join(dataset1_path, d))])

    # Copy folders from dataset1
    for folder in dataset1_ids:
        src = os.path.join(dataset1_path, folder)
        dest = os.path.join(output_path, folder)
        shutil.copytree(src, dest)

    # Find the next available ID in the output folder
    max_id = max(map(int, dataset1_ids)) if dataset1_ids else 0
    next_id = max_id + 1

    # Copy folders from dataset2
    dataset2_ids = sorted([d for d in os.listdir(dataset2_path) if os.path.isdir(os.path.join(dataset2_path, d))])
    id_mapping = {}  # Map old IDs to new IDs
    for folder in dataset2_ids:
        new_folder = str(next_id)
        id_mapping[int(folder)] = next_id
        next_id += 1

        src = os.path.join(dataset2_path, folder)
        dest = os.path.join(output_path, new_folder)
        shutil.copytree(src, dest)

    # Merge data.csv files
    data1 = pd.read_csv(os.path.join(dataset1_path, "data.csv"))
    data2 = pd.read_csv(os.path.join(dataset2_path, "data.csv"))

    # Offset IDs in data2 using the id_mapping
    data2["ID"] = data2["ID"].map(id_mapping)

    # Combine the two datasets
    combined_data = pd.concat([data1, data2], ignore_index=True)

    # Sort by ID
    combined_data = combined_data.sort_values(by="ID").reset_index(drop=True)

    # Save combined data.csv
    combined_data.to_csv(os.path.join(output_path, "data.csv"), index=False)
    print(f"Combined dataset saved to {output_path} and sorted by ID.")

def count_labels(data_csv_path):
    data = pd.read_csv(data_csv_path)

    label_counts = data["Label"].value_counts()

    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    return label_counts

# count_labels("rec_dataset2/data.csv")
count_avg_frames()
# combine_datasets_ordered_by_id("rec_dataset", "rec_dataset1", "rec_dataset2")
# create_subset()
# extract_data()
# copy_data()
# split_data()
# count_avg_frames()