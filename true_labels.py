import os
import cv2
import time
import math
import torch
import random
import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from threading import Thread
from pathlib import Path

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


def color_list():
    """给不同类别的框配置不同的颜色"""

    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = 1
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_xywh(path):
    try:
        with open(path, "r") as f:
            # 读取每一行label，并按空格划分数据
            label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            # return l
    except Exception as e:
        print("An error occurred while loading the file {}: {}".format(path, e))

    # print(label.shape)
    # 如果标注信息不为空的话
    if label.shape[0]:
        # 标签信息每行必须是五个值[class, x, y, w, h], 报出空标签，未标准化的标签也报出
        assert label.shape[1] == 5, "> 5 label columns: %s" % path
        assert (label >= 0).all(), "negative labels: %s" % path
        assert (label[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % path

    return label


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    """
    @param images: 图像 ==> 格式为：[batch_size, _, h, w]
    @param targets: 标签 ==> 格式：[batch_size, image_index, box_info]
    @param paths: 图像名字， 列表格式
    @param fname: 画图之后的保存名字，字符串路径
    @param names: 标签名
    @param max_size:
    @param max_subplots:
    @return:
    """
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 1
    tf = 1
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # colors = color_list()  # list of colors
    colors = [[219, 112, 147], [255, 20, 147], [218, 112, 214], [153, 50, 204], [65, 105, 225],
              [70, 130, 180], [102, 205, 170], [60, 179, 113], [32, 178, 170], [0, 139, 139],
              [184, 134, 11], [205, 133, 63], [210, 105, 30], [255, 69, 0], [178, 34, 34]]
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[int(cls)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def draw_prepare(label, label_dir, img_dir, class_names, save_dir, img_type):
    # 准备图像
    image_name = label.replace(".txt", img_type)  # 根据标签获取图像名字
    image_name_use = [image_name]  # 将图片名字转为勾画函数需要的格式 list
    # 确定图片路径，并检查
    image_path = os.path.join(img_dir, image_name)
    assert os.path.exists(image_path), "检查标签对应的图片路径是否正确?"
    # 读入图片并转为 numpy格式
    img = np.array(Image.open(image_path))  # [w, h, c]
    image = np.expand_dims(img, axis=0)  # 因为我想每次只画一张图，所以需要手动添加一个batch-size的维度: [w, h, c] ==> [batch_size, w, h, c]
    images = image.transpose((0, 3, 1, 2))  # 更改维度的位置，适配后面的输入: [batch_size, w, h, c] ==> [batch_size, c, w, h]
    # print(images.shape)  # 查看数据形状对不对

    # 读取标签并处理
    label_path = os.path.join(label_dir, label)
    labels = get_xywh(label_path)  # 根据图片路径读取标签，并进行简单检查
    # print('\n before:\n', labels)  # 检查标签
    # 配置勾画函数需要的标签的格式：[class, x, y, w, h] ==> [img_index, class, x, y, w, h]
    labels = np.insert(labels, 0, 0, axis=1)  # [需要插入的矩阵， 位置， 插入的内容， 在那个方向] 在每一行标签的 0 的位置上插入当前图像序号，由于我只画一张，所以就插入0
    # print("\n after:\n", labels)  # 检查标签

    # 准备保存名字
    save_name = os.path.join(save_dir, image_name)

    # 传入程序开始画图
    plot_images(images, labels, image_name_use, save_name, class_names)


def main():
    """修改这几个变量就可以了：image_dir, labels_dir, names, img_type"""
    image_dir = r'D:\python_project\YOLOv7-CTA\data\val\images'  # 图片存储文件夹
    labels_dir = r'D:\python_project\YOLOv7-CTA\data\val\labels'  # 标签存储文件夹
    image_box_save_dir = image_dir + "_box"  # 根据图片存放文件夹自动生成勾画box的文件夹
    if not os.path.exists(image_box_save_dir):
        os.makedirs(image_box_save_dir)

    # 类别名称 ==> 注意： 需要和 txt 文件里面的顺序对应，可以根据 PE_stage2.yaml 进行核对
    names = ["myeloblast", "promyelocyte", "N-myelocyte", "N-metamyelocyte", "N-band", "N-segmented", "E-myelocyte",
             "E-metamyelocyte", "E-band", "E-segmented", "basophil", "P-normoblast", "O-normoplast", "lymphocyte",
             "monocyte"]
    img_type = ".jpg"
    # -------------------------------------------------------------------------------------------------------------

    # 列出所有标签的的名字
    label_list = os.listdir(labels_dir)
    start = time.time()
    # 遍历标签名字，并根据标签名字找到对应的图片，将标签的里面的 bbox 画到图像上
    for label in tqdm(label_list):
        # 单线程
        draw_prepare(label, labels_dir, image_dir, names, image_box_save_dir, img_type)
        # 多线程
        # Thread(target=draw_prepare, args=(label, labels_dir, image_dir, names, image_box_save_dir, img_type)).start()

    end = time.time()
    time_use = end - start
    print(
        f"用时：{time_use // 3600} hour {(time_use - time_use // 3600) // 60} min {time_use - (time_use - time_use // 3600) // 60} seconds")

    print(f"\n勾画结果保存位置:   {image_box_save_dir}")


if __name__ == '__main__':
    main()
