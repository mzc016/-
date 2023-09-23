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
    weights_path = "save_weights/model20230923.pth"
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
            for boxes, classes in zip(predict_boxes, predict_classes):
                img_path_save.append(img_path)  # 始终是外层代表的当前图像
                pre_class_save.append(classes)
                pre_x1_save.append(boxes[0])
                pre_y1_save.append(boxes[1])
                pre_x2_save.append(boxes[2])
                pre_y2_save.append(boxes[3])


            plot_img = draw_objs(ori_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=3,
                                 font='simhei.ttf',
                                 font_size=20)
            # 保存预测的图片结果
            plot_img.save('./test_result/'+img_path + "_result.jpg")
        df = pd.DataFrame(
            {'image': img_path_save, 'label': pre_class_save, 'xmin': pre_x1_save, 'ymin': pre_y1_save, 'xmax': pre_x2_save, 'ymax': pre_y2_save})
        df.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()
