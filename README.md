# Mix-GCN.pytorch

## Result

| Method    | COCO    |VOC2007  |
|:---------:|:-------:|:--------:|
| ML-GCN |  83.0 |  94.0 |
| Ours(Mix-GCN+PPIR) |  84.3 |  95.6 |


### Requirements
Please, install the following packages
- numpy
- torch-0.3.1
- torchnet
- torchvision-0.2.0
- tqdm

### Download datasets
```sh
python3 download.py
```
### Download pretrain models
checkpoint/voc ([Dropbox](https://www.dropbox.com/s/m77n0tt6s7ewf0a/voc_checkpoint.pth.tar?dl=0))

checkpoint/coco ([Dropbox](https://www.dropbox.com/s/r4dwz5o13tbo0t9/coco_checkpoint.pth.tar?dl=0))

### Options
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

### Demo VOC 2007
```sh
CUDA_VISIBLE_DEVICES=1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 -e --resume checkpoint/voc/voc_checkpoint.pth.tar 
```

### Demo COCO 2014
```sh
CUDA_VISIBLE_DEVICES=1 python3 demo_coco_gcn.py data/coco/ --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar 
```

## Reference
This project is based on https://github.com/durandtibo/wildcat.pytorch

