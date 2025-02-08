import os
import shutil
import cv2
import numpy as np
import torch
from stack_3D_conv.preprocess_3d import preprocess_gpu, compute_optical_flow, normalize_img
from stack_3D_conv.two_stream_model import TemporalStream, SpatialStream, TwoStreamModel

device = "cuda" if torch.cuda.is_available() else "cpu"

CAMERA = 0
SPATIAL_STREAM_PATH = 'models/spatial_stream.pt'
TEMPORAL_STREAM_PATH = 'models/temporal_stream.pt'
MAX_FLOWS = 30
IMG_WIDTH = 224
IMG_HEIGHT = 224
OPTICAL_FLOW = cv2.cuda.FarnebackOpticalFlow.create(
    pyrScale = 0.4, numLevels = 4, winSize = 15,
    numIters = 5, polyN = 5, polySigma = 1.2, flags = 0
)
MAG_THRESHOLD = 1
LABELS = {
    0 : "Swipe Left", 
    1: "Swipe Right", 
    2: "Swipe Up", 
    3 : "Swipe Down", 
    4: "Zoom In",
    5: "Zoom Out",
    6: "Doing nothing"
}


def load_stream(spatial = False):
    stream_path = SPATIAL_STREAM_PATH if spatial else TEMPORAL_STREAM_PATH
    if not os.path.exists(stream_path):
        print("MODEL NOT FOUND! EXITING...")
        exit()

    stream = SpatialStream(num_classes=7) if spatial else TemporalStream(num_classes=7)
    checkpoint = torch.load(stream_path, weights_only=False)
    stream.load_state_dict(checkpoint['model_state_dict'])
    stream = stream.to(device)
    stream.eval()
    return stream

def load_two_stream_model():
    two_stream_model = TwoStreamModel(num_classes=7)
    
    spatial_stream_weights = torch.load(SPATIAL_STREAM_PATH, weights_only=False) 
    two_stream_model.spatial_stream.load_state_dict(spatial_stream_weights['model_state_dict'])

    temporal_stream_weights = torch.load(TEMPORAL_STREAM_PATH, weights_only=False)
    two_stream_model.temporal_stream.load_state_dict(temporal_stream_weights['model_state_dict'])

    two_stream_model = two_stream_model.to(device)
    return two_stream_model

def get_stream_prediction(model, input):
    with torch.no_grad():
        logits = model(input)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

        return logits, probabilities, confidence, predicted_class
    
def get_twostream_prediction(model, spatial_input, temporal_input):
    with torch.no_grad():
        logits = model(spatial_input, temporal_input)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

        return logits, probabilities, confidence, predicted_class
    

def run_loop(temporal_stream, spatial_stream, two_stream_model):
    
    #start camera and read input
    cap = cv2.VideoCapture(CAMERA)
    prev_gpu_frame = None
    flows = []
    counter = 0 
    frequency = 3
    label = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Missed a frame!")
            continue

        if len(flows) == MAX_FLOWS:
            flows.pop(0) # remove oldest frame
        
        curr_gpu_frame = preprocess_gpu(frame, IMG_HEIGHT, IMG_WIDTH)

        #at beginning of loop
        if prev_gpu_frame is None:
            prev_gpu_frame = curr_gpu_frame
            continue
        
        #append latest optical flow vecs
        flow_vectors = compute_optical_flow(OPTICAL_FLOW, MAG_THRESHOLD, prev_gpu_frame, curr_gpu_frame)
        flows.append(flow_vectors)

        counter += 1

        displayed_img = frame
        if len(flows) == MAX_FLOWS and counter % frequency == 0:
            
            #SPATIAL STREAM -----------------------------------------------------------------------------
            #use normalized last frame at 224x224 for spatial stream
            frame_bgr = cv2.resize(frame, (IMG_WIDTH,IMG_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            normalized_frame_rgb = normalize_img(frame_rgb)
            last_frame_tensor = torch.tensor(normalized_frame_rgb, dtype=torch.float32).unsqueeze(0).to(device)
            last_frame_tensor = last_frame_tensor.permute(0,3,1,2)
            
            # #get model prediction
            # logits, probabilities, confidence, predicted_class = get_stream_prediction(spatial_stream, last_frame_tensor)
            # spatial_label = LABELS[int(predicted_class)]
            # print(f"[SPATIAL] Prediction: {spatial_label} with confidence {float(confidence):.2f}")

            #TEMPORAL STREAM ----------------------------------------------------------------------------
            flow_tensor = torch.tensor(np.stack(flows), dtype=torch.float32).unsqueeze(0).to(device)
            flow_tensor = flow_tensor.permute(0,2,1,3,4) #(batch_size = 1, 2, 30, 224, 224)

            # #get model prediction
            # logits, probabilities, confidence, predicted_class = get_stream_prediction(temporal_stream, flow_tensor)
            # print(f"[TEMPORAL] LOGITS       : {logits.cpu().numpy().tolist()}")
            # print(f"[TEMPORAL] PROBABILITIES: {probabilities.cpu().numpy().tolist()}")
            # temporal_label = LABELS[int(predicted_class)]
            # print(f"[TEMPORAL] Prediction: {temporal_label} with confidence {float(confidence):.2f}")


            #TWO STREAM ----------------------------------------------------------------------------------
            logits, probabilities, confidence, predicted_class = get_twostream_prediction(two_stream_model, last_frame_tensor, flow_tensor)
            
            if confidence > 0.90 or (label == "Doing nothing" and confidence > 0.40):
                label = LABELS[int(predicted_class)].upper()
                print(f"[TWO-STREAM] Prediction: {label} with confidence {float(confidence):.2f}") 
            counter = 0

        cv2.putText(displayed_img, text = label, org=(30, 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=3)
        #set current frame as previous
        prev_gpu_frame = curr_gpu_frame

        #display
        cv2.imshow("Action recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    
if __name__ == "__main__":

    temporal_stream = load_stream(spatial=False)
    spatial_stream = load_stream(spatial=True)
    two_stream_model = load_two_stream_model()
    run_loop(temporal_stream, spatial_stream, two_stream_model)
    exit(1)
