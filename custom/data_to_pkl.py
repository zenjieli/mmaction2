import os

import json
import glob
import shutil
import pickle

import numpy as np

data_path = 'data/cauca_kul'

# Load json file
with open('/home/ms/david/fall_detection/data.json') as f:
    data = json.load(f)

# Create pkl data
pkl_data = {'split':{'train':os.listdir(os.path.join(data_path, 'train')), 'test':os.listdir(os.path.join(data_path, 'test'))}, 'annotations':[]}
images = []
for split in ['train', 'test']:
    for clip in data[split]:
        data_dict = {}
        data_dict['frame_dir'] = split+'_'+clip
        data_dict['label'] = data[split][clip]['label']
        if '_' not in data[split][clip]['paths'][0]:
            data_dict['img_shape'] = (480, 720)
            keypoints_path = '/home/ms/david/fall_detection/data/CAUCAFall/keypoints/predictions/'
        else:
            data_dict['img_shape'] = (480, 800)
            keypoints_path = '/home/ms/david/fall_detection/data/KUL/keypoints/predictions/'
        keypoints = []      
        keypoint_score = []
        for frame in data[split][clip]['paths']:
            with open(os.path.join(keypoints_path, frame.replace('png', 'json'))) as file:
                k_data = json.load(file)[0]
            keypoints.append(k_data['keypoints'])
            keypoint_score.append(k_data['keypoint_scores'])


        data_dict['keypoint'] = np.array([keypoints])
        data_dict['keypoint_score'] = np.array([keypoint_score])
        data_dict['original_shape'] = data_dict['img_shape']
        data_dict['total_frames'] = 10
        pkl_data['annotations'].append(data_dict)

with open('cauca_kul.pkl', 'wb') as pkl:
    pickle.dump(pkl_data, pkl)

