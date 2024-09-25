import torch
from mmaction.apis import init_recognizer, inference_recognizer

# config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
config_file = 'configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py'
checkpoint_file = 'cfgs/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint_20220815-17eaa484.pth'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, checkpoint_file, device=device)
# inference the demo video
input_video = 'data/hmdb51/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0'
# res = inference_recognizer(model, 'demo/demo.mp4')
results = inference_recognizer(model, input_video)

# show the results
# labels = open('tools/data/ucf101/label_map.txt').readlines()
# labels = [x.strip() for x in labels]
# results = [(labels[k[0]], k[1]) for k in results]

# print(f'The top-5 labels with corresponding scores are:')
# for result in results:
#     print(f'{result[0]}: ', result[1])