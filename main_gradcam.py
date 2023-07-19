import os
import random
import time
import argparse
import numpy as np
from models.gradcam import YOLOV7GradCAM
from models.yolov7_object_detector import YOLOV7TorchObjectDetector
import cv2

# 数据集中的类别名(与标签数字相对应)
names = ["myeloblast", "promyelocyte", "N-myelocyte", "N-metamyelocyte", "N-band", "N-segmented", "E-myelocyte",
         "E-metamyelocyte", "E-band", "E-segmented", "basophil", "P-normoblast", "O-normoplast", "lymphocyte", "monocyte"]  # class names
# yolov7网络中，detect层前的三层输出
target_layers = ['102_act']  # yolov7
# target_layers = ['105_act']  # yolov7-CTA

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="runs/train/exp_v7-1-1/weights/best.pt", help='Path to the model')
parser.add_argument('--img-path', type=str, default='data/test/images', help='input image path')
parser.add_argument('--output-dir', type=str, default='runs/test/outputs/', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default='76_act',
                    help='The layer hierarchical address to which gradcam will applied, the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam method: gradcam, gradcampp')
parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None, help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
parser.add_argument('--no_text_box', action='store_true',  help='do not show label and box on the heatmap')
args = parser.parse_args()


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    img = cv2.imread('temp.jpg')

    # # Plots one bounding box on image img
    # tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # tl = 1
    # color = color or [random.randint(0, 255) for _ in range(3)]
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # if label:
    #     tf = 1
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


# 检测单个图片
def main(img_path):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    device = args.device
    input_size = (args.img_size, args.img_size)
    # 读入图片
    img = cv2.imread(img_path)  # 读取图像格式：BGR
    print('[INFO] Loading the model')
    # 实例化YOLOv7模型，得到检测结果
    model = YOLOV7TorchObjectDetector(args.model_path, device, img_size=input_size, names=names)
    # img[..., ::-1]: BGR --> RGB
    # (480, 640, 3) --> (1, 3, 480, 640)
    torch_img = model.preprocessing(img[..., ::-1])
    tic = time.time()
    # 遍历三层检测层
    for target_layer in target_layers:
        # 获取grad-cam方法
        if args.method == 'gradcam':
            saliency_method = YOLOV7GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        # elif args.method == 'gradcampp':
        #     saliency_method = YOLOV7GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  # 得到预测结果
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        # 保存设置
        imgae_name = os.path.basename(img_path)  # 获取图片名
        save_path = f'{args.output_dir}{imgae_name[:-4]}/{args.method}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f'[INFO] Saving the final image at {save_path}')
        # 遍历每张图片中的每个目标
        for i, mask in enumerate(masks):
            # 遍历图片中的每个目标
            res_img = result.copy()
            # 获取目标的位置和类别信息
            bbox, cls_name = boxes[0][i], class_names[0][i]
            label = f'{cls_name} {conf[0][i]}'  # 类别+置信分数
            # 获取目标的热力图
            res_img, heat_map = get_res_img(bbox, mask, res_img)
            res_img = plot_one_box(bbox, res_img, label=label, color=colors[int(names.index(cls_name))], line_thickness=3)
            # 缩放到原图片大小
            res_img = cv2.resize(res_img, dsize=(img.shape[:-1][::-1]))
            output_path = f'{save_path}/{target_layer[:-4]}_{i}.jpg'
            cv2.imwrite(output_path, res_img)
            print(f'{imgae_name[:-4]}_{target_layer[:-4]}_{i}.jpg done!!')
    print(f'Total time : {round(time.time() - tic, 4)} s')


if __name__ == '__main__':
    # 图片路径为文件夹
    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        print(img_list)
        for item in img_list:
            # 依次获取文件夹中的图片名，组合成图片的路径
            main(os.path.join(args.img_path, item))
    # 单个图片
    else:
        main(args.img_path)
