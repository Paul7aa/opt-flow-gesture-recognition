import cv2
import numpy as np
import os
import time
from stack_3D_conv.preprocess_3d import get_aug_params, apply_all_augs

IMG_HEIGHT = 224
IMG_WIDTH = 224
FRAMES = 20

def apply_rotation_aug(frame, angle):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)  #center of the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  #rotation matrix
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return rotated_frame

def apply_brightness_aug(frame, brightness_factor):
    frame = frame.astype(np.float32)
    frame = np.clip(frame * brightness_factor, 0, 255)
    # Convert back to uint8 for display
    return frame.astype(np.uint8)

def apply_contrast_aug(frame, contrast_factor):
    frame = frame.astype(np.float32)
    frame = np.clip(frame * contrast_factor, 0, 255)
    return frame.astype(np.uint8)

def apply_scaling_aug(frame, scale_factor):
        height, width = frame.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        if scale_factor < 1.0:
            #pad if scaling down
            padded_frame = np.zeros_like(frame)
            y_offset = (frame.shape[0] - new_height) // 2
            x_offset = (frame.shape[1] - new_width) // 2
            padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = scaled_frame
            return padded_frame
        else:
            #crop if scaling up
            y_offset = (new_height - frame.shape[0]) // 2
            x_offset = (new_width - frame.shape[1]) // 2
            return scaled_frame[y_offset:y_offset + frame.shape[0], x_offset:x_offset + frame.shape[1]]

def dense_optical_flow(folder_path, magnitude_threshold, augment):
    #measure time
    start_time = time.time()

    frame_files = sorted(os.listdir(folder_path))
    
    if len(frame_files) < 2:
        print("Error: Need at least two frames to compute optical flow.")
        return
    
    #read the first frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]))

    if prev_frame is None:
        print(f"Error: Unable to read the first frame: {frame_files[0]}")
        return
    
    aug_params = None
    if augment:
        aug_params = get_aug_params()
        print(f"AUG PARAMS: {aug_params}")
        prev_frame = apply_all_augs(prev_frame, aug_params, optical_flow=True)

    #upload to gpu
    prev_gpu_frame = cv2.cuda.GpuMat()
    prev_gpu_frame.upload(prev_frame)

    #convert to grayscale
    # prev_frame = cv2.blur(prev_frame, (3,3))
    prev_gpu_frame = cv2.cuda.resize(prev_gpu_frame, (IMG_WIDTH, IMG_HEIGHT))
    prev_gpu_frame = cv2.cuda.cvtColor(prev_gpu_frame, cv2.COLOR_BGR2GRAY)

    stacked_vectors = np.zeros((FRAMES * 2, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    #create cuda optical flow instance
    optical_flow = cv2.cuda.FarnebackOpticalFlow.create(
        numLevels = 4, pyrScale=0.4, fastPyramids=False, winSize=15,
        numIters=5, polyN=5, polySigma=1.2, flags=0
    )

    # tvl1 = cv2.cuda.OpticalFlowDual_TVL1.create(
    #     tau = 0.25,
    #     lambda_= 0.3,
    #     theta = 0.3,
    #     nscales = 4,
    #     warps = 4,
    #     epsilon = 0.01,
    #     iterations = 30,
    #     scaleStep = 0.5,
    #     gamma = 0.0)

    valid_vectors = 0
    for i in range(1, len(frame_files)):
        # Read the next frame
        curr_frame = cv2.imread(os.path.join(folder_path, frame_files[i]))
        # curr_frame = cv2.blur(curr_frame, (3,3))
        if curr_frame is None:
            print(f"Error: Unable to read frame: {frame_files[i]}")
            continue
    
        #apply aug
        if augment:
            curr_frame = apply_all_augs(curr_frame, aug_params, optical_flow=True)

        #load frame to gpu
        curr_gpu_frame = cv2.cuda.GpuMat()
        curr_gpu_frame.upload(curr_frame)
        curr_gpu_frame = cv2.cuda.resize(curr_gpu_frame, (IMG_WIDTH, IMG_HEIGHT))
        curr_gpu_frame = cv2.cuda.cvtColor(curr_gpu_frame, cv2.COLOR_BGR2GRAY)

        gpu_flow = optical_flow.calc(prev_gpu_frame, curr_gpu_frame, None)
        flow = gpu_flow.download()

        #compute magnitude and angle of the flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        #filter out small motions
        mask = magnitude > magnitude_threshold
        filtered_magnitude = magnitude * mask
        filtered_angle = angle * mask

        #if no significant movement, ignore vector
        if np.count_nonzero(filtered_magnitude) == 0:
            continue

        if valid_vectors >= FRAMES:
            break
        stacked_vectors[2 * valid_vectors, :, :] = cv2.normalize(filtered_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        stacked_vectors[2 * valid_vectors + 1 , :, :] = cv2.normalize(filtered_angle, None, 0, 1, cv2.NORM_MINMAX)
        valid_vectors += 1

        hsv = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        hsv[..., 0] = (filtered_angle * 180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr_representation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


        # Display the optical flow visualization
        cv2.imshow('Filtered Dense Optical Flow', bgr_representation)
        viz_gpu_frame = curr_gpu_frame.download()
        cv2.imshow('Current Frame', viz_gpu_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        # Update the previous frame
        prev_gpu_frame = curr_gpu_frame.clone()

    end_time = time.time()
    print("TIME ELAPSED FOR OPTICAL FLOW COMPUTATION: ", end_time - start_time)

# examples
i = 0
for i in range (700,1050):
    folder_path = f"rec_dataset2/{i}"
    print(folder_path)
    dense_optical_flow(folder_path, 1, True)
 

# import pandas as pd
# df = pd.read_csv("rec_dataset2/data.csv")
# for folder in  sorted(os.listdir("rec_dataset2/")):
#     if "00_" in folder or ".csv" in folder:
#         continue
#     row = df[df['ID'] == int(folder)]

#     if not row.empty:
#         label = row['Label'].values[0]
#         if label == "Zoom in" or label == "Zoom out":
#             files = sorted(os.listdir(f"rec_dataset2/{folder}"))
#             last_file = files[-1]
#             for file in files:
#                 print(f"rec_dataset2/{folder}/{file}")
#                 frame = cv2.imread(f"rec_dataset2/{folder}/{file}")

#                 cv2.imshow(f"{int(folder)} - {label}", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#                 if file == last_file:
#                     cv2.waitKey()
#                     cv2.destroyAllWindows()



# IMG_WIDTH = 176
# IMG_HEIGHT= 100
# for folder_path in sorted(os.listdir("jester_subset/data")):
#     print(folder_path)
#     dense_optical_flow(f"jester_subset/data/{folder_path}")
