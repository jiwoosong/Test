
# 동영상 초해상화 네트워크

### 1. 코드 구성
---
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
---
#### 폴더 구성
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

#### 실행 파일 (train.py) 세팅
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
```model_name```는 ```video_module/basicsr/models/archs/```에 구현된 각 모듈의 class 이름으로 string 형태로 불러오도록 구성됨.  ```crop```사이즈는 인접 영역 탐색 크기이며 인접 영역 탐사를 수행하지 않는 모델의 경우 0을 넣음. ```pretrain_network_g```는 트레이닝 시작시 초기 설정되는 weight의 경로를 설정하도록 되어있음.

```opt_input```의 경우 트레이닝에서 설정한 옵션으로 본 연구에서 활용한 옵션은 ```video_super_resolution/video_module/TRAINING/option/``` 안에 ```1e-4.yml``` 파일임. 
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










#### Charades_v1_480_loader
```bash
Charades_v1_480_img
  ├──0A8CF
    ├──0000000.jpg
    ├──0000001.jpg
    ...
  ├──0A8ZT
  ...
```
#### HumanEva
```bash
HumanEva
  ├──image
    ├──S1_Box_1_(C1)
      ├──000000.jpg
      ├──000001.jpg
    ├──S1_Box_1_(C2)
    ...
  ├──video
    ├──S1
      ├──Box_1_(C1).avi
      ├──Box_1_(C2).avi
      ...
    ├──S2
    ...
  ├──test_list.csv, train_list.csv, val_list.csv
```
폴더는 위 구성을 따름. 
Charades 데이터 세트는 ```Charades_v1_480_img``` 폴더 안에 각 동영상 클립 이름으로 폴더로 구성하며 각 프레임은 ```000000.jpg```로 시작함.
HumanEva 데이터 세트는 ```HumanEva``` 안에 ```image```,```video``` 폴더로 구성되며 각 jpg와 avi 파일로 구성됨.

폴더를 구성하고 Charades와 HumanEva 데이터세트의 경로를 반영하기 위해서는 ```video_restoration/video_module/Total_Train_*.py``` 파일에서 loader의 ```root_dir```을 수정하면 경로를 인식할 수 있음.



### 3. 트레이닝 설정
---
```bash
training
  ├──resnet
    ├──pixsim_1by1
      ├──Results
      ├──main.py
    ├──pixsim_1by1_flat2d_stride1
    ├──pixsim_1by1_flat2d_stride2
  ├──unet
    ├──Skipconnection
    ├──Embedding
    ├──Spatial_weight
    ├──Flat
```
```unet``` 폴더 내부는 skip connection 연결에 따른 실험 ```Skip Connection```, FC/Conv 임베딩 구조 개선 실험 ```Embedding```, 유사도 검출 및 공간 가중합산 실험 ```Spatial_weight```, 축소 계층 실험 ```Flat``` 실험으로 구성되어있음. 이때 pixsim 모델은 Weight 단일 가중합, pixwise 모델은 공간 가중합 모델임. 
```resnet```폴더 내부는 action recognition 정보를 활용하기 위한  인코더 디코더 구조 개선실험이 첨부됨.
각 실험에 대한 결과는 ```Results``` 에 저장되어있으며 함께 있는 python 파일은 실험을 돌리기 위한 main.py이다. 이때, main.py는 다음과 같이 구성되어 있음.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #GPU

# Define Test Param
input_dic={}
input_dic['test_type'] = 'nextframe' # 'nextframe', 'subtract', 'opticalflow'
input_dic['wsupervised'] = 0
input_dic['QSsep'] = False
input_dic['load'] = False

# Dataset
input_dic['dataset'] = 'Humaneva'
input_dic['frame_interval'] = 50
input_dic['random_interval'] = False
input_dic['batch_size'] = 1
input_dic['W'] = 64
input_dic['H'] = 64
input_dic['input_size'] = 10 # Senetences size

# Enc / Dec
from video_module.Network_Module.Unet.Unet_conv import Unet_Up
from video_module.Network_Module.Unet.Unet_conv import Unet_Down
channels = [3, 32, 64, 128]
noskip = [True,True,True]
input_dic['encoder'] = Unet_Down(channels=channels, skip_ind=noskip)
input_dic['decoder'] = Unet_Up(channels=channels, skip_ind=noskip)

# Memory
from video_module.Memory_Module.Static.unet.Static_Conv_multi_sep1by1_pixsim import mem
Mem_param={}
Mem_param['hops'] = 1
Mem_param['input_size'] = input_dic['input_size'] # Senetences size
Mem_param['embed_size'] = 128 # Unet/Resnet's last dimension
Mem_param['memory_size']  = 10 # Memory size
Mem_param['pad']  = 3
Mem_param['noskip'] = noskip
Mem_param['channels'] = channels
input_dic['mem_net'] = mem(Mem_param).cuda()

# Visdom
input_dic['vis_port'] = 7097
# Training
input_dic['max_iter'] = 500000 #iter
input_dic['val_iter'] = 100 #iter
input_dic['save_iter'] = 1000 #iter


# Define optimizer/lr/loss
import torch
optimList = [{"params": input_dic['encoder'].parameters(), "lr": 1e-3},
             {"params": input_dic['decoder'].parameters(), "lr": 1e-3},
             {"params": input_dic['mem_net'].parameters(), "lr": 1e-3}]
input_dic['optim'] = torch.optim.Adam(optimList)#,weight_decay=1e-5)
input_dic['criterion'] = torch.nn.L1Loss()

# Start Training
from video_module.Total_Prepare import Prepare
from video_module.Total_Train_unet import training
input_param = Prepare(input_dic)
training(input_dic)

```

실행파일은 dictionary ```input_dic```에서 설정한 옵션에 따라 동작됨. 
다양한 옵션이 있는것을 확인할 수 있지만 실제 최종보고서에서 실험한 세팅과 관련없는 옵션들은 꺼두었음.
```input_dic```과 기타 조정할 수 있는 옵션으로는 다음이 있음.

```input_dic['dataset']``` : 데이터세트 종류 (Humaneva, Charades_v1_480)

```input_dic['frame_interval']``` : 프레임 간격 (defalut=50)

```input_dic['W']```,```input_dic['H']``` : 이미지 크기 (default=64)

```input_dic['vis_port']``` : Visdom 포트 번호 (defalut=7097)

```input_dic['max_iter']``` : 학습 Iteration

```input_dic['val_iter']``` : 평가 Iteration

```input_dic['val_iter']``` : 저장할 Iteration

```input_dic['optim']``` : Optimizer (default=Adam)

```input_dic['criterion']``` : 손실함수 (default=L1)

```Mem_param['embed_size']``` : 메모리네트워크에 저장될 채널 사이즈

#### 기타 옵션

```channels, noskip``` : 인코더/디코더의 각 채널과 스킵커넥션 연결 구성




### 4. 저장된 결과
---
```training``` 각 실험의 ```Results``` 폴더 또는 새롭게 학습시켜 얻은 ```Results```폴더는 다음 구성을 따름. 

(일부 폴더 구성이 다른 경우가 있을 수 있음.)
```bash
Results
  ├──nextframe_fi50_Humaneva # 테스트이름_프레임간격_데이터세트 종류
    ├──Static_Conv_multi_sep1by1_m10_e128_hop1 # 사용한 메모리 네트워크 이름
      ├──unet_conv[3,32,64,128][True, True, True] # 인코더 디코더 네트워크 이름
        ├──L1Loss_e1e-03_d1e-03_W64H64 # 손실함수_러닝레이트_이미지 
          ├──best_loss
            ├──out_image
              ├──0
                ├──*.jpg
              ├──1
              ...
            ├──weight_image
              ├──0
                ├──*.jpg
              ├──1
              ...
          ├──last
            ├──out_image
            ├──weight_image
              ...
          ├──weight
            ├──best_loss.pth
            ├──last.pth
```

```best_loss```는 학습 중간 가장 최적의 validation loss를 달성했을때의 validation 결과를 저장한 것이고, ```last```는 매 ```input_dic['save_iter']```마다 저장한 결과임.
이때 validation 결과로 출력된 결과는 ```out_image```에 저장되며, 각 계층의 weight을 시각화 한 그림은```weight_image```로 저장됨.
