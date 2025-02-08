import cv2
import numpy as np
import os

def dense_optical_flow(folder_path, img_height, img_width, farneback_params, magnitude_thresh):
    
    frame_files = sorted(os.listdir(folder_path))
    
    if len(frame_files) < 2:
        print("Error: Need at least two frames to compute optical flow.")
        return
    
    #read the first frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]))
    if prev_frame is None:
        print(f"Error: Unable to read the first frame: {frame_files[0]}")
        return
    
    #upload to gpu
    prev_gpu_frame = cv2.cuda.GpuMat()
    prev_gpu_frame.upload(prev_frame)

    #convert to grayscale
    # prev_frame = cv2.blur(prev_frame, (3,3))
    prev_gpu_frame = cv2.cuda.resize(prev_gpu_frame, (img_width, img_height))
    prev_gpu_frame = cv2.cuda.cvtColor(prev_gpu_frame, cv2.COLOR_BGR2GRAY)
    
    magnitude_cumulated = np.zeros((img_height, img_width), dtype=np.float32)
    direction_cumulated = np.zeros((img_height, img_width), dtype=np.float32)
    
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

    for i in range(1, len(frame_files)):
        # Read the next frame
        curr_frame = cv2.imread(os.path.join(folder_path, frame_files[i]))

        # curr_frame = cv2.blur(curr_frame, (3,3))
        if curr_frame is None:
            print(f"Error: Unable to read frame: {frame_files[i]}")
            continue
        
        curr_gpu_frame = cv2.cuda.GpuMat()
        curr_gpu_frame.upload(curr_frame)
        curr_gpu_frame = cv2.cuda.cvtColor(curr_gpu_frame, cv2.COLOR_BGR2GRAY)
        curr_gpu_frame = cv2.cuda.resize(curr_gpu_frame, (img_width, img_height))

        #compute dense optical flow
        gpu_flow = optical_flow.calc(prev_gpu_frame, curr_gpu_frame, None)
        flow = gpu_flow.download()

        #compute magnitude and angle of the flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        #filter out small motions
        mask = magnitude > magnitude_thresh
        filtered_magnitude = magnitude * mask
        filtered_angle = angle * mask

        #accumulate magnitude and direction
        magnitude_cumulated += filtered_magnitude
        direction_cumulated += filtered_angle * filtered_magnitude

        # Update the previous frame
        prev_gpu_frame = curr_gpu_frame.clone()

    # Normalize cumulative magnitude and average direction
    magnitude_normalized = cv2.normalize(magnitude_cumulated, None, 0, 1, cv2.NORM_MINMAX)
    
    average_direction = np.divide(
        direction_cumulated,
        magnitude_cumulated,  # Avoid division by zero
        out=np.zeros_like(direction_cumulated),
        where=magnitude_cumulated > 0
    )
    direction_normalized = average_direction / (2 * np.pi)

    result = np.array([magnitude_normalized, direction_normalized])

    return result
    
    

    