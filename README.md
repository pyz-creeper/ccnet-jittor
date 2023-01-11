# ccnet-jittor

This is a [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) implementation of [CCNet](https://arxiv.org/abs/1811.11721) for semantic segmentation.

## Quick Start

### Environments

You should install Jittor under the guidance of [Jittor official](https://cg.cs.tsinghua.edu.cn/jittor/download/).

### Training

To train with default batch-size 16 setting, you only need to specify the desired log directory, and run: 

```
mpirun -np 4 python train.py --log_dir <log directory>
```

### Testing

To test a model, you can specify the model checkpoint directory and the model's backbone, and run:

```
python test.py --ckpt_dir <log directory> --model_backbone <van or resnet>
```

### Inference

To visualize a result, you can specify the picture's directory and the model, and run:

```
python visualize.py --ckpt_dir <log directory> --model_backbone <van or resnet> --pic_dir <picture directory> --save_dir <save directory>
```

### Pretrained Checkpoints

#### Pretrained Backbone

| Backbone    | Checkpoints                                                  |
| ----------- | ------------------------------------------------------------ |
| ResNet101-C | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/42156484364446a7843f/) |
| VAN-base    | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/180a73bb9cb44daf9a68/) |



#### CCNet Results

| Settings                                | mIoU                        | Checkpoints                                                  |
| --------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| ResNet101-C, batch-size 16              | 41.72 (42.80 wi multiscale) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/facbf00602b44be2908b/) |
| ResNet101-C, batch-size 8               | 38.85                       | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/80e98339aac8434397ca/) |
| ResNet101-C, batch-size 8, dilated      | 38.72                       | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/651763215b37480cae51/) |
| ResNet101-C, batch-size 8, neighborhood | 36.77                       | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/a0a3316b36a6459295bd/) |
| VAN-base, batch-size 16                 | 36.82                       | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/902923e7821847a7ae12/) |

