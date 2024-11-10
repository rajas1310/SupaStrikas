# Imports
import os
import sys

sys.path.append('FootballPassPrediction/PerspectiveTransform')

from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import numpy as np
from clustimage import Clustimage
from matplotlib import image
import matplotlib.pyplot as plt
from collections import Counter
import cv2

from elements.perspective_transform import Perspective_Transform
from elements.assets import transform_matrix, detect_color
from elements.args import Args

import torch
import math
from pathlib import Path
import traceback as tb
from gnn.gnn_models import *
from  gnn.utils import *
import gnn.gnn_datagen as ggen

import importlib
importlib.reload(ggen)

# Build and load model (Yolov8l without pretraining, 1080 resolution)
model = YOLO('/content/drive/MyDrive/hacksc2024/best_60epochs.pt')

# device = 0 #0,1,2=cuda device 0,1,2 else 'cpu'

video_path = '/content/drive/MyDrive/hacksc2024/Clip_5.mp4'
out_video_path = f'/content/drive/MyDrive/hacksc2024/Output/{Path(video_path).stem}_out.mp4'

#Load Arguments for perspective transform model
opt = Args()

# Load model
pt_model = Perspective_Transform(opt)


def yolo_inf(video_path):
    results = model(video_path)
    detect_df = [['pred_class', 'pred_x', 'pred_y', 'pred_w', 'pred_h', 'image_name', 'OID']] #Headers

    for frame,res in enumerate(results): #Iterate results of prediction
        max_ball_conf = 0.
        max_ball_conf_row = None
        for i, box in enumerate(res.boxes):
            oid = i+1
            coords = box.xywhn.detach().cpu()
            cls = box.cls.detach().cpu()
            cls = int(cls.data[0])
            conf = box.conf.detach().cpu()
            conf = float(conf.data[0])
            if cls == 1 and conf > max_ball_conf: #Get max conf ball out of all ball predictions for image
                max_ball_conf = conf
                max_ball_conf_row = [cls ,coords[0][0].item(), coords[0][1].item(), coords[0][2].item(), coords[0][3].item(), frame, oid]
            elif cls == 0: #For player, just directly add the prediction
                row = [cls ,coords[0][0].item(), coords[0][1].item(), coords[0][2].item(), coords[0][3].item(), frame, oid]
                detect_df.append(row)
        if max_ball_conf_row is not None:
            detect_df.append(max_ball_conf_row)

    headers = detect_df.pop(0)
    detect_df = pd.DataFrame(detect_df, columns=headers)
    
    groupdf=detect_df.groupby("image_name")
    
    ndetect=[]
    dropped=[]
    for name,df in groupdf:
        if len(df[df["pred_class"]==1])==1:
            ndetect.append(df)
        else:
            dropped.append(name)
    ndetect_df=pd.concat(ndetect)
    ndetect_df=ndetect_df.dropna()
    
    return ndetect_df

def pre_process_crop(image, bbox):
    H = image.shape[0]
    W = image.shape[1]

    x = int(bbox[0]*W)
    y = int(bbox[1]*H)
    w = int(bbox[2]*W)
    h = int(bbox[3]*H)

    #Extracting Crop from given image with bbox specs
    crop = image[y:y+h, x:x+w]
    crop = cv2.resize(crop, (20,40))

    #Applying filters to the crop
    crop = cv2.medianBlur(crop,3)
    crop = cv2.bilateralFilter(crop,5,30,30)

    return crop

def formatPred(pred):
    y = np.unique(pred, return_counts=True)
    z = sorted(zip(y[1], y[0]), reverse=True)
    res = pred.copy()
    i=0
    for tup in z:
        idx_list = np.where(pred == tup[1])[0]

    # print(tup, idx_list)
        for idx in idx_list:
            res[idx] = i
        i+=1

    return res

# Clustering function for team identification
def cluster(yolo_df):
    total_boxes = 0
    total_misclassified = 0

    groups = yolo_df.groupby("image_name")

    vid_path = video_path
    cap = cv2.VideoCapture(vid_path)

  # loop through each group and apply the code snippet to each image
    for image_name, group in tqdm(groups):
        frame_id=int(image_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = frame
        players_info = group[group['pred_class'] == 0]

        min_num_cats = 3
        max_num_cats = 4

        player_boxes = []

        for index, row in players_info.iterrows():

            bbox = [row['pred_x']-(row['pred_w']/2), row['pred_y']-(row['pred_h']/2), row['pred_w'], row['pred_h']]
            crop = pre_process_crop(img, bbox)
            player_boxes.append(crop)

        player_boxes_reshaped = np.array(player_boxes).reshape(len(player_boxes),-1)
        player_boxes_reshaped = player_boxes_reshaped/255.0

        cl = Clustimage(method='pca',
                    embedding='tsne',
                    grayscale=False,
                    dim=(20,40),
                    params_pca={'n_components':0.95},
                    verbose=60)

        results = cl.fit_transform(player_boxes_reshaped,
                                cluster='agglomerative',
                                evaluate='silhouette',
                                metric='euclidean',
                                linkage='ward',
                                min_clust=min_num_cats,
                                max_clust=max_num_cats,
                                cluster_space='high')

        pred_labels = results['labels']

        preds = formatPred(pred_labels)

        i = 0
        for index, row in players_info.iterrows():
            if(int(row['pred_class']) != 0):
                continue
            yolo_df.loc[index, 'pred_team'] = int(preds[i])
            i+=1
        print("name: {}, preds: {}, total boxes in : {}".format(image_name, preds, players_info.shape[0]))
        total_boxes += players_info.shape[0]

    print("total_boxes: {} ,  total_images: {}".format(total_boxes, len(groups)))

    return yolo_df

def generate_perspective_transform(model, image_path, yolo_df):

    groups = yolo_df.groupby("image_name")
    vid_path = video_path
    capt = cv2.VideoCapture(vid_path)
    # loop through each group and apply the code snippet to each image
    for image_name, group in tqdm(groups):
        frame_id=int(image_name)
        capt.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = capt.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap = frame

        w = cap.shape[1] #1280
        h = cap.shape[0] #720

        main_frame = cap.copy()

        M, warped_image = model.homography_matrix(main_frame)


        for index, row in group.iterrows():
            if(math.isnan(row['pred_class'])):
                continue
            else:
                x_center = row['pred_x'] #if x is center
                y_center = row['pred_y'] + (row['pred_h']/2)

                coords = transform_matrix(M, (x_center*w, y_center*h), (h, w), (68, 105))

                yolo_df.loc[index, 'gnd_x'] = coords[0]
                yolo_df.loc[index, 'gnd_y'] = coords[1]

    return yolo_df

# Write final predictions from dataframe to video for visualization
# import imutils

def draw_boxes_on_frame(frame, boxes):
    for box in boxes:
        color = (0, 0, 255) if box['bp'] == 1 else (0, 255, 0)
        x, y, w, h = box['pred_x'], box['pred_y'], box['pred_w'], box['pred_h']
        x1, y1 = int((x - w / 2) * frame.shape[1]), int((y - h / 2) * frame.shape[0])
        x2, y2 = int((x + w / 2) * frame.shape[1]), int((y + h / 2) * frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        attributes = []
        if box['bp'] == 1:
            attributes.append('possession')
        if box['br'] == 1:
            attributes.append('receiver')
        attributes_str = ', '.join(attributes)
        cv2.putText(frame, attributes_str, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    bp_boxes = [box for box in boxes if box['bp'] == 1]
    br_boxes = [box for box in boxes if box['br'] == 1]
    if bp_boxes and br_boxes:
        bp_box = bp_boxes[0]
        br_box = br_boxes[0]
        bp_x, bp_y, bp_w, bp_h = bp_box['pred_x'], bp_box['pred_y'], bp_box['pred_w'], bp_box['pred_h']
        br_x, br_y, br_w, br_h = br_box['pred_x'], br_box['pred_y'], br_box['pred_w'], br_box['pred_h']
        bp_center_x, bp_center_y = int(bp_x * frame.shape[1]), int(bp_y * frame.shape[0])
        br_center_x, br_center_y = int(br_x * frame.shape[1]), int(br_y * frame.shape[0])
        arrow_color = (255, 255, 255)
        cv2.arrowedLine(frame, (bp_center_x, bp_center_y), (br_center_x, br_center_y), arrow_color, 3)

    return frame


def Ball_Possession(transformed_df):
    def calc_distance(row, ballx, bally):
        x2 = row['gnd_x']
        y2 = row['gnd_y']
        distance = np.sqrt((ballx - x2)**2 + (bally - y2)**2)
        return distance

    transformed_df['gnd_x'] = transformed_df['gnd_x'].apply(lambda x: max(x, 0))
    transformed_df['gnd_y'] = transformed_df['gnd_y'].apply(lambda x: max(x, 0))

    transformed_df['bp']=0
    transformed_df['distance']=0

    grouped_df=transformed_df.groupby('image_name')


    for name,group in grouped_df:

        row = group.loc[group['pred_class'] == 1, ['gnd_x', 'gnd_y']].iloc[0]
        ballx=row['gnd_x']
        bally=row['gnd_y']
        group['distance'] = group.apply(calc_distance, args=(ballx, bally), axis=1)

        ball_idx= group.index[group['pred_class'] == 1]
        group.loc[ball_idx,'distance'] =  np.inf

        mask = group['pred_team'] > 1
        ppl_idx = group[mask].index
        group.loc[ppl_idx,'distance'] =  np.inf

        min_index = group['distance'].idxmin()
        group.loc[min_index, 'bp'] = 1
        group.loc[ball_idx,'distance'] = 0

        transformed_df.update(group)
        #transformed_df.loc[transformed_df["image_name"]==name]= group
        return transformed_df

def gnn_processing(bp_df):
    bp_df['br']=0
    file_path = "./processed/data.pt"
    if os.path.exists(file_path):
        os.remove(file_path)


    graph_list=ggen.csv2data(csv_df="df2",inp_df=bp_df,label_smoothing=0.0,graphfunc=2,undirected=False,negEdge=True)
    dataset=MyDataset("./",graph_list)

    val_loader = DataLoader(dataset, batch_size=1)

    model = GAT2(hidden_channels= [32,32,32],heads=5,dropout=0)
    PATH = './gnn/model_weights/best_model_GATv2_pre_ntrain_5.pt'
    state = torch.load(PATH)
    model.load_state_dict(state['state_dict'])

    for val_data in val_loader:
        output = model(val_data)
        pred_br= torch.argmax(output)
        bp_df.loc[(bp_df['image_name'] == float(val_data.name[0])) & (bp_df['OID'] == val_data.oid[0][pred_br]), 'br'] = 1
    
    return bp_df

# Helper function for main processing pipeline
def process_video(video_path: str, out_video_path: str) -> str:
    # Run YOLO inference
    ndetect_df = yolo_inf(video_path)
    clustered_df = cluster(ndetect_df)
    
    try:
        with torch.no_grad():
            transformed_df = generate_perspective_transform(pt_model, None, clustered_df)
    except Exception as e:
        print("Error during perspective transformation:", e)
        return "Error in processing"

    bp_df = Ball_Possession(transformed_df)
    updated_bp_df = gnn_processing(bp_df)

    # Process video frame by frame
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_boxes = updated_bp_df[updated_bp_df['image_name'] == frame_number]
        bp_boxes = frame_boxes[frame_boxes['bp'] == 1]
        br_boxes = frame_boxes[frame_boxes['br'] == 1]

        combined_boxes = pd.concat([bp_boxes, br_boxes], ignore_index=True, sort=False)
        
        if not combined_boxes.empty:
            frame = draw_boxes_on_frame(frame, combined_boxes.to_dict('records'))

        out.write(frame)

    cap.release()
    out.release()
    
    return out_video_path


    
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_utils import FirebaseStorageSingleton
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase
cred = credentials.Certificate("cerebralhacks-dba8c-firebase-adminsdk-tebp7-ae6980afe0.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'cerebralhacks-dba8c.appspot.com'
})


app = FastAPI(title="Video Generation API",debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific origins if needed
    allow_credentials=False,
    allow_methods=["*"],  # Adjust this if you want to restrict methods
    allow_headers=["*"],  # Adjust this if you want to restrict headers
)

from pydantic import BaseModel
class VideoResponse(BaseModel):
    video_url: str

# FastAPI endpoint to process video
@app.post("/process-video/")
async def process_video_endpoint(file: str):
    # Create a reference to the video file
    bucket = storage.bucket()
    blob = bucket.blob(file)
    blob.download_to_filename('scene.mp4')

    # Save uploaded video temporarily
    temp_video_path = 'scene.mp4'
    # with video_path(delete=False, suffix=".mp4") as tmp:
    #     temp_video_path = tmp.name
    #     tmp.write(await file.read())

    output_video_path = f"./output/{Path(file).stem}_processed.mp4"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Run the main process
    result_path = process_video(temp_video_path, output_video_path)
    
    firebase_storage = FirebaseStorageSingleton()
    bucket = firebase_storage.get_bucket()

    try:
        # Upload the file
        destination_blob_name = result_path.split('/')[-1]
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(result_path)
        print(f"File {'scene.mp4'} uploaded to {destination_blob_name}.")
    except Exception as e:
        print(f"An error occurred while uploading the file: {str(e)}")

    try:
    # Generate signed URL
        expires_at = datetime.now() + timedelta(days=7)  # URL expires in 7 days
        signed_url = blob.generate_signed_url(expires_at)
        print(f"Signed URL: {signed_url}")
        
        return VideoResponse(video_url=signed_url)
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)