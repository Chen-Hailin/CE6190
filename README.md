# Exps results on Segmenter: Transformer for Semantic Segmentation

## Installation
For this method, we use cuda 11.1 and A100 GPUs. To install
```sh
export DATASET=/path/to/dataset/dir
```

```sh
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install .
python -c "import mmseg;import os;print(os.path.dirname(mmseg.__file__)+'/datasets')" | xargs -I {} cp segm/data/mmseg/* {}
```

## Train

Train on games dataset:
```python
CUDA_VISIBLE_DEVICES=0 python -m segm.train --log-dir seg_small_mask_games --dataset games --backbone vit_small_patch16_384 --decoder mask_transformer --port=12340 2>&1 | tee 002.log
```
Edit `segm/config.yml` to perform ablation on hyper-parameters. Change `--decoder mask_transformer` to `--decoder linear` to do ablation on decoder model. 

## Inference
Output predictions on validation data
```python
python -m segm.inference --model-path seg_small_mask_games/checkpoint.pth -i ../dataset/games/images/val/ -o seg_small_mask_games/games_pred 
```
The output files are in `seg_small_mask_games/games_pred`

To evaluate on ADE20K, run the command:
```python
# single-scale evaluation:
python -m segm.eval.miou seg_small_mask_games/checkpoint.pth games --singlescale
# multi-scale evaluation:
python -m segm.eval.miou seg_small_mask_games/checkpoint.pth games  --multiscale
```


## Citation

```
@article{strudel2021,
  title={Segmenter: Transformer for Semantic Segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2105.05633},
  year={2021}
}
```
