import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
import pandas as pd
import numpy as np


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=5)

    # load train weights
    weights_path = "save_weights/resNetFpn-model-29.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    img_path_list = os.listdir('./test')
    ori_img_list = []
    img_list = []
    data_transform = transforms.Compose([transforms.ToTensor()])
    for img_path in img_path_list:
        temp = Image.open('./test/'+img_path)
        ori_img_list.append(temp)
        temp = data_transform(temp)
        temp = torch.unsqueeze(temp, dim=0)
        img_list.append(temp)

    #  下面几个save为构造dataframe准备数组
    img_path_save = []
    pre_class_save = []
    pre_x1_save = []
    pre_y1_save = []
    pre_x2_save = []
    pre_y2_save = []
    pre_sc_save = []

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init  启动模型
        img_height, img_width = img_list[0].shape[-2:]  # 随便取一张，反正就是启动模型而已
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        for img_path, img, ori_img in zip(img_path_list, img_list, ori_img_list):
            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
            need_sort = []
            for boxes, classes ,scores in zip(predict_boxes, predict_classes, predict_scores):
                if scores>0.5:
                    need_sort.append([img_path, classes, boxes[0], boxes[1], boxes[2], boxes[3], scores])
            need_sort.sort(key=lambda x: (x[2], x[3]))
            img_path_save.extend([row[0] for row in need_sort])  # 始终是外层代表的当前图像
            pre_class_save.extend([row[1]-1 for row in need_sort])  # 训练中1-4对应目标，0对应背景，最终结果0对应吊臂
            pre_x1_save.extend([row[2] for row in need_sort])
            pre_y1_save.extend([row[3] for row in need_sort])
            pre_x2_save.extend([row[4] for row in need_sort])
            pre_y2_save.extend([row[5] for row in need_sort])
            pre_sc_save.extend([row[6] for row in need_sort])

            # 这里重复构造一下 predict_boxes和predict_class，首先是经过了筛选，另一方面是验证我上面排序过的框是否正确、
            predict_boxes = [row[2:6] for row in need_sort]
            predict_classes = [row[1] for row in need_sort]
            predict_scores = [row[6] for row in need_sort]
            '''
            "吊臂": 1,
            "吊钩": 2,
            "吊臂下的人": 3,
            "其他人": 4
            '''
            for i in range(len(predict_boxes)):
                if predict_classes[i] == 3 or predict_classes == 4:  # 是一个人
                    predict_classes[i] = 4  # 首先默认是其他人
                    if 1 in predict_classes:  # 图中有吊臂
                        for j in range(len(predict_boxes)):  # 就开始循环找图中的吊臂，和这个人的位置做匹配
                            if predict_classes[j] == 1:  # 找到吊臂
                                left = max(predict_boxes[i][0], predict_boxes[j][0])  # 相交的话左端点就是两个xmin的最大值
                                right = min(predict_boxes[i][2], predict_boxes[j][2])  # 右端点是xmax的最小值
                                if left < right:
                                    predict_classes[i] = 3
                                    break  # 在寻找吊臂的循环中，找到一个头顶的吊臂了



            plot_img = draw_objs(ori_img,
                                 np.array(predict_boxes),
                                 np.array(predict_classes),
                                 np.array(predict_scores),
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=3,
                                 font='simhei.ttf',
                                 font_size=20)
            # 保存预测的图片结果
            plot_img.save('./test_result/'+img_path + "_result.jpg")

        df = pd.DataFrame(
            {'image': img_path_save, 'label': pre_class_save, 'xmin': pre_x1_save, 'ymin': pre_y1_save, 'xmax': pre_x2_save, 'ymax': pre_y2_save, 'score':pre_sc_save})
        df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
