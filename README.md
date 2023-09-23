
## 文件结构：
```
  ├── backbone: 特征提取网络
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块
  ├── my_dataset.py: 自定义dataset用于读取数据集
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重下载地址（下载后放入backbone文件夹中）：
* MobileNetV2 weights(下载后重命名为`mobilenet_v2.pth`，然后放到`bakcbone`文件夹下): https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* Resnet50 weights(下载后重命名为`resnet50.pth`，然后放到`bakcbone`文件夹下): https://download.pytorch.org/models/resnet50-0676ba61.pth
* ResNet50+FPN weights: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是`fasterrcnn_resnet50_fpn_coco.pth`文件，
  不是`fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`，然后放到当前项目根目录下即可。
 

## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本

## 注意事项
* 在使用训练脚本时，注意要将`--data-path`(VOC_root)设置为自己存放`VOCdevkit`文件夹所在的**根目录**
* 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer，
  即不传递norm_layer变量，默认去使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。




## Faster RCNN框架图
![Faster R-CNN](fasterRCNN.png) 
