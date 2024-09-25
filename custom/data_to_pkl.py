import os
import json
import pickle
import numpy as np

# All the path
data_path = 'data/cauca_kul'
json_path = '/home/ms/david/fall_detection/data.json'

# Load json file
with open(json_path) as f:
    data = json.load(f)

# Create pkl data
pkl_data = {'split':{'train':os.listdir(os.path.join(data_path, 'train')), 'test':os.listdir(os.path.join(data_path, 'test'))}, 'annotations':[]}
images = []
for split in ['train', 'test']:
    for clip in data[split]:
        # For each clip
        data_dict = {}
        data_dict['frame_dir'] = split+'_'+clip
        data_dict['label'] = data[split][clip]['label']

        # If it belongs to CAUCAFall, else KU Leuven, get the keypoints path
        if '_' not in data[split][clip]['paths'][0]:
            data_dict['img_shape'] = (480, 720)
            keypoints_path = '/home/ms/david/fall_detection/data/CAUCAFall/keypoints/predictions/'
        else:
            data_dict['img_shape'] = (480, 800)
            keypoints_path = '/home/ms/david/fall_detection/data/KUL/keypoints/predictions/'

        # Open json file created by RTMPose
        keypoints = []      
        keypoint_score = []
        for frame in data[split][clip]['paths']:
            with open(os.path.join(keypoints_path, frame.replace('png', 'json'))) as file:
                k_data = json.load(file)[0]
            keypoints.append(k_data['keypoints'])
            keypoint_score.append(k_data['keypoint_scores'])

        # Other attributes required by mmaction2
        data_dict['keypoint'] = np.array([keypoints])
        data_dict['keypoint_score'] = np.array([keypoint_score])
        data_dict['original_shape'] = data_dict['img_shape']
        data_dict['total_frames'] = 10
    
        pkl_data['annotations'].append(data_dict)

# Export the file
with open('cauca_kul.pkl', 'wb') as pkl:
    pickle.dump(pkl_data, pkl)

