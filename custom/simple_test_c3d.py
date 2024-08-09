import torch
from mmaction.apis import init_recognizer, inference_recognizer

# config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
config_file = 'configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py'
checkpoint_file = 'work_dirs/c3d_sports1m_16x1x1_45e_ucf101_split_0_rgb/best_top1_acc_epoch_5.pth'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, checkpoint_file, device=device)
# inference the demo video
input_video = 'data/ucf101/rawframes/ApplyLipstick/v_ApplyLipstick_g01_c02'
# res = inference_recognizer(model, 'demo/demo.mp4')
results = inference_recognizer(model, input_video)

# show the results
labels = open('tools/data/ucf101/label_map.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])