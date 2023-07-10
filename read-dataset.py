# -*- coding: utf-8 -*-
import shutil
import os


def objFileName(file_name_list):
    obj_name_list = []
    for i in open(file_name_list, 'r'):
        obj_name_list.append(i.replace('\n', ''))
    return obj_name_list


def copy_file(path, file_name_list):
    # 指定存放图片的目录
    for i in objFileName(file_name_list):
        new_obj_name = i
        dir, file = os.path.split(new_obj_name)
        shutil.copy(new_obj_name, path + '/' + file)


if __name__ == '__main__':
    # train_images_name_list = "D:/python_project/YOLOv7-CTA/VOC_marrow_cells/train.txt"
    train_labels_name_list = "D:/python_project/YOLOv7-CTA/VOC_marrow_cells/train_labels.txt"
    # train_images_path = r'D:/python_project/YOLOv7-CTA/data/train/images/'
    train_labels_path = r'D:/python_project/YOLOv7-CTA/data/train/labels/'
    # val_images_name_list = "D:/python_project/YOLOv7-CTA/VOC_marrow_cells/val.txt"
    val_labels_name_list = "D:/python_project/YOLOv7-CTA/VOC_marrow_cells/val_labels.txt"
    # val_images_path = r'D:/python_project/YOLOv7-CTA/data/val/images/'
    val_labels_path = r'D:/python_project/YOLOv7-CTA/data/val/labels/'
    # copy_file(train_images_path, train_images_name_list)
    copy_file(train_labels_path, train_labels_name_list)
    # copy_file(val_images_path, val_images_name_list)
    copy_file(val_labels_path, val_labels_name_list)
