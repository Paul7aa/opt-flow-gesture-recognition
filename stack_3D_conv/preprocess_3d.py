import cv2
import numpy as np
import os
import random

def get_aug_params():
    rotate_angle = random.uniform(-15, 15)
    scale_factor = random.uniform(0.8, 1.2)
    brightness_factor = random.uniform(0.7, 1.3)
    contrast_factor = random.uniform(0.7, 1.3)
    hue_factor = random.uniform(-0.1, 0.1)
    saturation_factor = random.uniform(0.7, 1.3)

    return rotate_angle, scale_factor, brightness_factor, contrast_factor, hue_factor, saturation_factor
    
def apply_rotation_aug(frame, angle):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)  #center of the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  #rotation matrix
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return rotated_frame
    
def apply_brightness_aug(frame, brightness_factor):
    frame = frame.astype(np.float32)
    frame = np.clip(frame * brightness_factor, 0, 255)
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
    
def apply_hue_saturation_aug(frame, hue_factor, saturation_factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_factor * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0, 255)
    adjusted_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted_frame


def apply_all_augs(frame, params, optical_flow):
    rotate_angle, scale_factor, brightness_factor, contrast_factor, hue_factor, saturation_factor = params
    frame = apply_rotation_aug(frame, rotate_angle)
    frame = apply_scaling_aug(frame, scale_factor)
    frame = apply_brightness_aug(frame, brightness_factor)
    frame = apply_contrast_aug(frame, contrast_factor)
    if not optical_flow:
        frame = apply_hue_saturation_aug(frame, hue_factor, saturation_factor)

    return frame

def interpolate_flow(flow1, flow2, alpha):
    return (1 - alpha) * flow1 + alpha * flow2

def process_stacked_vectors(stacked_vectors, max_flows):

    if len(stacked_vectors) == max_flows:
        return np.array(stacked_vectors)

    original_len = len(stacked_vectors)

    if original_len < max_flows:
        num_interpolations = max_flows - original_len
        intervals = original_len - 1
        interpolations_per_interval = [num_interpolations // intervals] * intervals

        for i in range(num_interpolations % intervals):
            interpolations_per_interval[i] += 1
            
        new_stacked_vectors = []
        for i in range(intervals):
            new_stacked_vectors.append(stacked_vectors[i])  #add original frame
            #add interpolated frames
            for interp in range(interpolations_per_interval[i]):
                alpha = (interp + 1) / (interpolations_per_interval[i] + 1)
                interpolated_frame = interpolate_flow(stacked_vectors[i], stacked_vectors[i + 1], alpha)
                new_stacked_vectors.append(interpolated_frame)
        new_stacked_vectors.append(stacked_vectors[-1])  #add last frame

        return np.array(new_stacked_vectors[:max_flows])
    
    else:
        indices = np.linspace(0, original_len - 1, max_flows, dtype=int)
        stacked_vectors = np.array(stacked_vectors) 
        return stacked_vectors[indices]

def normalize_img(img_rgb):
    # use image net mean and std
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    for c in range(3):
        img_rgb[c, :, :] = (img_rgb[c, :, :] - mean[c]) / std[c]
    
    return img_rgb

def get_random_img(folder_path, img_height, img_width, augment):
    files = os.listdir(folder_path)
    filename = random.choice(files)
    img_path = os.path.join(folder_path, filename)
    
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (img_height, img_width))
    #BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if augment:
        #apply augmentations
        aug_params = get_aug_params()
        img_rgb = apply_all_augs(img_rgb, aug_params, optical_flow=False)

    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    img_rgb = normalize_img(img_rgb)    
    
    return img_rgb

def get_last_img(folder_path):
    files = os.listdir(folder_path)
    file_idx = len(files) - 1
    img_path = os.path.join(folder_path, files[file_idx])
    
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))  #channels first

    #normalize channels with imagenet std and mean
    img_rgb = normalize_img(img_rgb)

    return img_rgb

def compute_optical_flow(optical_flow, magnitude_thresh, prev_gpu_frame, curr_gpu_frame):
    gpu_flow = optical_flow.calc(prev_gpu_frame, curr_gpu_frame, None)
    flow = gpu_flow.download()

    dx, dy = flow[..., 0], flow[..., 1]
    
    #filter out small motions
    mask = (dx ** 2 + dy ** 2) > magnitude_thresh**2
    filtered_dx = dx * mask
    filtered_dy = dy * mask
    
    min_dx, max_dx = filtered_dx.min(), filtered_dx.max()
    min_dy, max_dy = filtered_dy.min(), filtered_dy.max()

    range_dx = max_dx - min_dx
    range_dy = max_dy - min_dy

    if range_dx > 0:
        normalized_dx = 2 * (filtered_dx - min_dx) / range_dx - 1
    else:
        normalized_dx = filtered_dx

    if range_dy > 0:
        normalized_dy = 2 * (filtered_dy - min_dy) / range_dy - 1
    else:
        normalized_dy = filtered_dy

    return np.stack([normalized_dx, normalized_dy])

def preprocess_gpu(frame, img_width, img_height):
    gpu_frame = cv2.cuda.GpuMat()
    gpu_frame.upload(frame)
    gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
    gpu_frame = cv2.cuda.resize(gpu_frame, (img_width, img_height))

    return gpu_frame


def compute_vectors_from_folder(folder_path, img_height, img_width, max_flows, farneback_params, magnitude_thresh, augment):
    
    frame_files = sorted(os.listdir(folder_path))
    n_frames = len(frame_files)

    raw_frames = max_flows + 1

    #picking random max_flows window
    if n_frames < raw_frames:
    #not enough frames to pick random window
        start_idx = 0
        end_idx = n_frames
    elif n_frames == raw_frames:
        start_idx = 0
        end_idx = n_frames
    else:
        start_idx = random.randint(0, n_frames - raw_frames)
        end_idx = start_idx + raw_frames

    #read the first frame
    first_frame_path = os.path.join(folder_path, frame_files[start_idx])
    prev_frame = cv2.imread(first_frame_path)
    
    
    if prev_frame is None:
        print(f"Error: Unable to read the first frame: {frame_files[start_idx]}")
        return
    
    aug_params = None
    if augment:
        aug_params = get_aug_params()
        prev_frame = apply_all_augs(prev_frame, aug_params, optical_flow=True)
    
    #upload to gpu
    prev_gpu_frame = preprocess_gpu(prev_frame, img_width, img_height)
    
    #create cuda optical flow instance
    optical_flow = cv2.cuda.FarnebackOpticalFlow.create(
        pyrScale = farneback_params["pyr_scale"],
        numLevels = farneback_params["levels"],
        winSize = farneback_params["winsize"],
        numIters = farneback_params["iterations"],
        polyN = farneback_params["poly_n"],
        polySigma = farneback_params["poly_sigma"],
        flags = farneback_params["flags"],
    )

    
    stacked_vectors = []
    for i in range(start_idx + 1, end_idx):
        #read the next frame
        curr_frame_path = os.path.join(folder_path, frame_files[i])
        curr_frame = cv2.imread(curr_frame_path)

        if curr_frame is None:
            print(f"Error: Unable to read frame: {frame_files[i]}")
            continue
        
        #apply aug
        if augment:
            curr_frame = apply_all_augs(curr_frame, aug_params, optical_flow=True)

        curr_gpu_frame = preprocess_gpu(curr_frame, img_width, img_height)

        #compute dense optical flow
        stacked_flow_vec = compute_optical_flow(optical_flow, magnitude_thresh, prev_gpu_frame, curr_gpu_frame)

        stacked_vectors.append(stacked_flow_vec)
        #update the previous frame
        prev_gpu_frame = curr_gpu_frame.clone()

    #interpolates if less than max_flows optical flow frames were computed
    stacked_vectors = process_stacked_vectors(stacked_vectors=stacked_vectors, max_flows=max_flows)
    stacked_vectors = np.transpose(stacked_vectors, (1,0,2,3)) #channels first(dx/dy), temporal dimension

    return stacked_vectors

if __name__ == "__main__":
    result = compute_vectors_from_folder("rec_dataset/1", 224, 224, 30, farneback_params= {
            "pyr_scale": 0.4,
            "levels": 4,
            "winsize": 15,
            "iterations": 5,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0 },
            magnitude_thresh=1,
            augment=False)
    
    print(result.shape)
        

    