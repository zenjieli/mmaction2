==Installation==

* Create new new environment
```
conda create --name mm python=3.8 -y
conda activate mm
```
* Install Pytorch
```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```
* Install *mmlab* libraries
```
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmpose
```
* Install *mmactions*
Enter the `mmaction` directory.
```
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
```

==Datasets==

When running `resize_videos.py`, a wrong version of `ffmpeg` in the current environment may cause the error "Unknown encoder 'libx264'". The fix is to run `/usr/local/ffmpeg` explictily in the script.

==Configuration==

* Explanation on `SampleFrames`: <https://github.com/open-mmlab/mmaction2/issues/1662>