import cv2
import os
import keyboard
import pandas as pd
import time
from collections import Counter


def read_dataframe(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Dataframe not found, creating a new one")
        df = pd.DataFrame(columns=["ID", "Label"])
        return df


def save_dataframe(output_directory, df):
    csv_path = os.path.join(output_directory, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


def warmup_camera(cap, duration=2):
    print(f"Warming up the camera for {duration} seconds...")
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from the camera during warmup!")
            break
        cv2.imshow("Camera Warmup", frame)
        cv2.waitKey(1)


def record_gesture(cap, output_folder, gesture_label, dataframe):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    folders = sorted(os.listdir(output_folder))
    if folders:
        folder_numbers = [int(folder) for folder in folders if folder.isdigit()]
        next_folder_number = max(folder_numbers) + 1 if folder_numbers else 1
    else:
        next_folder_number = 1

    new_folder_path = os.path.join(output_folder, str(next_folder_number))
    os.makedirs(new_folder_path)

    frame_count = 0
    print("Recording video. Press any key to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        cv2.imshow("Recording", frame)

        # Save frame
        frame_filename = os.path.join(new_folder_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        print(f"Saved: {frame_filename}")
        if cv2.waitKey(1) != -1:
            print("Stopping recording.")
            break

    new_row = pd.DataFrame({"ID": [next_folder_number], "Label": [gesture_label]})
    dataframe = pd.concat([dataframe, new_row], ignore_index=True)
    print(f"Recording complete. Total frames saved: {frame_count}")
    return dataframe

def count_class_distribution_csv(csv_path):
    df = pd.read_csv(csv_path)
    class_counts = Counter(df['Label'])
    return class_counts

gestures = {
    "1": "Swipe left",
    "2": "Swipe right",
    "3": "Swipe Up",
    "4": "Swipe Down",
    "5": "Zoom in",
    "6": "Zoom out",
    "7": "Doing nothing"
}

if __name__ == "__main__":
    output_folder = "rec_dataset2"
    class_distribution = count_class_distribution_csv("rec_dataset2/data.csv")
    print(class_distribution)

    df = read_dataframe(os.path.join(output_folder, "data.csv"))

    
    # Initialize the camera once
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera!")
        exit(1)

    try:
        warmup_camera(cap, duration=2)

        while True:
            print(
                '''Select gesture to record:
                1. Swipe left
                2. Swipe right
                3. Swipe Up
                4. Swipe Down
                5. Zoom in
                6. Zoom out
                7. Doing nothing'''
            )

            key = keyboard.read_event(suppress=True)
            if key.event_type == "down":
                if key.name in gestures:
                    gesture = gestures[key.name]
                    print("Gesture: ", gesture)
                    df = record_gesture(cap, output_folder, gesture_label=gesture, dataframe=df)
                else:
                    save_dataframe(output_folder, df)
                    break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_dataframe(output_folder, df)
