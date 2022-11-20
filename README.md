# Exps on SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

## Installation

In our working environment, we use cuda 11.0 version with A100 GPUs. To install:
```
conda create --name segformer python=3.8
source activate segformer
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
pip install -e . --user
```

## Training
Download `weights` 
(
[google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) | 
[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ)
) 
pretrained on ImageNet-1K, and put them in a folder ```pretrained/```.

### Single-gpu training & evaluation
```
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.768x768.games.57k.py 1 12340 2>&1 | tee 002.log
```
change `B2/segformer.b2.768x768.games.57k.py` to `B3/segformer.b3.768x768.games.57k.py` | `B1/segformer.b1.768x768.games.57k.py` | `B2/segformer.b2.768x768.games.57k.nopretrain.py` for encoder ablation.
change it to `B2/segformer.b2.768x768.games.57k.bs2.py` | `B2/segformer.b2.768x768.games.57k.bs6.py` for hyper-parameter ablation (batch size in this case)


## Inference


```
python tools/test.py local_configs/segformer/B2/segformer.b2.768x768.games.57k.py ./work_dirs/segformer.b2.768x768.games.57k/latest.pth
```
Default directory to save the predicted images are `./games_pred`; to change it, edit `--show-dir` argument in `tools/test.py`


## License
Please check the LICENSE file. SegFormer may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
