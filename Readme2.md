
# 동영상 초해상화 네트워크

### 1. 코드 구성

#### - 코드 준비
---
첨부된 코드는 EDVR에서 제공하는 코드의 구조를 일부 따름. 제공된 코드는 Anaconda 가상환경을 활용하며 아래 설치를 수행하도록 함.

```bash
activate pytorch #"Conda Env Name"
conda install future lmdb matplotlib numpy Pillow pyyaml requests scikit-image scipy tqdm yapf
conda install ninja==1.9 pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install addict tb-nightly opencv-python==3.4.11.43 opencv-contrib-python==3.4.11.43
```
이때 EDVR의 PCD Alignment 모듈에서 작동되는 deforamable convolution의 경우 C 컴파일이 필요함. 따라서 EDVR git 프로젝트를 빌드하고 빌드된 deformable convolution을 가져오도록 함.

```bash
git clone https://github.com/xinntao/BasicSR.git
python setup.py develop
```
빌드 이후 ```EDVR/basicsr/models/ops/dcn/``` 에 빌드된 ```deform_conv_ext.*.so``` 파일을 제공된 코드의 ```video_super_resolution/video_module/EDVR_UTILS/basicsr/models/ops/dcn/```로 이동하면 코드를 구동하기 위한 기본적인 준비를 완료할 수 .

#### - 코드 구조
---
```bash
video_super_resolution
  ├──video_module
    ├──EDVR_UTILS
    ├──Memory_Module
    ├──TESTING
    ├──TRAINING
    ├──..etc
```

EDVR_UTILS      : EDVR에서 제공하는 유틸리티 중 일부를 모아놓은 폴더. ```basicsr```은 전체 EDVR 네트워크를 구성하기 위한 모듈들이 구현되어 있으며 연구에 활용한 네트워크는 ```video_super_resolution/video_module/EDVR_UTILS/basicsr/models/archs```에 구현되어있음.

Memory_Module   : 메모리 네트워크가 구현되어있는 폴더.

TRAINING/TESTING  : 학습 / 평가를 수행하기 위한 유틸리티를 모아놓은 폴더.





### 2. 데이터세트 설정
---
연구에서 활용한 데이터세트는 REDS 데이터세트로 2019 NTIRE 챌린지에서 공개되었으며 SNU 서버 또는 Google Drive를 통해 다운로드 받을 수 있다. 다운로드를 위한 링크는 https://seungjunnah.github.io/Datasets/reds.html 에서 다운로드 받을 수 있다.
```bash
REDS
  ├──train_sharp_bicubic
  ├──train_sharp
```
학습에 활용되는 동영상 클립의 갯수는 270개로 training 세트 266개 validation 세트 4개로 구성됨. 단, ```meta_info_REDS_GT.txt```와 ```meta_info_REDS4_test_GT.txt```로 트레이닝과 평가 세트를 구분하기 때문에 하나의 폴더에 구분없이 ```0~269``` 동영상 클립이 들어있는 폴더를 구성하도록 함.





### 3. 트레이닝 설정

#### - 폴더 구성
---
본 연구에서 구성한 실험 세팅은 ```TRAINING``` 폴더 내부에 ```train1```,```train2```, ```train3```에 정리되어있음.
```bash
Training
  ├──option
  ├──train1
    ├──experiments
      ├──EDVR_Final... # Model Name
        ├──models
          ├──*.pth # weight
    ├──*.py
  ├──train2
  ├──train3
  ├──...etc
```
```train1``` : 최종 보고서 Table 1에 도시된 실험 세팅이 구성되어있음. ```pixsim_*.py```, ```pixswise_*.py```, ```pixsentence_*.py```는 각각 단일 가중치 합성 메모리, 공간 가중치 합성 메모리, 전체공간 탐색 모델임.

```train2``` : Table 2에 도시된 실험 세팅. ```256_concat.py``` , ```128_concat_norm.py```은 각각 "채널 256 concat crop 3x3"과 "채널 256 concat 정규화 crop 3x3" 모델의 실험 세팅임. ```128_crop3.py``` 는 128채널 모델임. 128채널 5x5 모델은 ```train3```의 PCD 있는 모델과 동일함.

```train3``` : Table 3에 도시된 실험 세팅. 내부 폴더는 ```train3/PCD```, ```train3/noPCD``` 모델로 구성하여 각 Table 3 실험으로 구성되어있음.

#### - 실행 파일 세팅
---
각 ```train1```,```train2```,```train3``` 내부에는 실행 ```*.py```들이 있으며 세팅할 수 있는 옵션은 다음과 같음.
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from video_module.TRAINING import TRAIN_LOAD_FREEZE, TRAIN_LOAD_SCRATCH

TRAIN_LOAD_FREEZE.TRAIN(model_name = '',
                        opt_input = '',
                        resume_state = None,
                        proj_name = '',
                        crop = 3,
                        pretrain_netwrok_g = None,
                        )
```
```model_name```는 ```video_module/basicsr/models/archs/```에 구현된 각 모듈의 class 이름으로 string 형태로 불러오도록 구성됨.  ```crop```사이즈는 인접 영역 탐색 크기이며 인접 영역 탐사를 수행하지 않는 모델의 경우 0을 넣음. ```pretrain_network_g```는 트레이닝 시작시 초기 설정되는 weight의 경로를 설정하도록 되어있음. ```opt_input```은 학습에 활용될 옵션 ```.yml```파일로 아래 설명으로 연결됨.

#### - 옵션 파일 세팅
---
```opt_input```의 경우 트레이닝에서 설정한 옵션으로 본 연구에서 활용한 옵션파일은 ```video_super_resolution/video_module/TRAINING/option/``` 안에 ```1e-4.yml``` 임. 
옵션은 크게 ```datasets```, ```network_g```, ```path```, ```train```, ```val```로 구성되어 있음.

```datasets``` : 데이터세트의 경로, 로더를 설정할 수 있음. ```train```과 ```val```에 대해 ```dataroot_gt```, ```dataroot_lq```, ```meta_info_file```의 경로를 프로젝트의 절대 경로로 변경하여 수정하도록 함.

```yml
# dataset and data loader settings
datasets:
  train:
    name: REDS
    type: REDSDataset
    dataroot_gt: /home/pub/db/REDS/train_sharp/train_sharp/ # path to ground truth
    dataroot_lq: /home/pub/db/REDS/train_sharp_bicubic/X4/ # path to input image
    dataroot_flow: ~
    meta_info_file: /home/adrenaline36/바탕화면/Jiwoo_Work/ETRI/ETRI_FINAL/JW/EDVR_UTILS/meta_info/meta_info_REDS_GT.txt # path to train/val index text file (just like .csv)
    val_partition: REDS4  # set to 'official' when use the official validation partition
    io_backend:
      type: disk
    num_frame: 5 # Adjacent frame
    gt_size: 256 # SR Output shape
    interval_list: [1] # Frame Interval
    random_reverse: false # Augmentations
    use_flip: false # Augmentations
    use_rot: true # Augmentations
    # data loader
    use_shuffle: true
    num_worker_per_gpu: 1
    batch_size_per_gpu: 4
    dataset_enlarge_ratio: 200
    prefetch_mode: ~

  val:
    name: REDS4
    type: VideoTestDataset
    dataroot_gt: /home/pub/db/REDS/val/train_sharp/
    dataroot_lq: /home/pub/db/REDS/val/train_sharp_bicubic/
    meta_info_file: /home/adrenaline36/바탕화면/Jiwoo_Work/ETRI/ETRI_FINAL/JW/EDVR_UTILS/meta_info/meta_info_REDS4_test10_GT.txt
    # change to 'meta_info_REDSofficial4_test_GT' when use the official validation partition
    io_backend:
      type: disk
    cache_data: false
    num_frame: 5
    padding: reflection_circle
```

이외,

```network_g``` : EDVR 네트워크의 구조를 정의할 수 있는 부분으로 EDVR 세팅으로 따로 변경할 필요는 없음.

```path``` : pre_train 네트워크를 불러오거나 학습을 재개할 수 있는 weight을 연결하는 부분으로 따로 변경할 필요는 없음.

```train``` : train phase에서 필요한 optimizer 세팅 ```optim_g```, schelduler 세팅 ```scheduler```, iteration 등 을 수정할 수 있음. ```tsa_iter```는 다른 네트워크가 동결되고 TSA만 학습되는 iteration을 말함.

```val``` : validation phase에서 필요한 평가 방법(PSNR/SSIM) ```metrics```, 평가 iteration 등을 설정할 수 있음.


### 4. 평가 설정
---
#### - 폴더 구성
















