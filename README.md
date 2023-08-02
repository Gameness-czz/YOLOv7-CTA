# Improved YOLOv7 Algorithm for Detecting Bone Marrow Cell

Zhizhao Cheng and Yuanyuan Li

## Introduction
In this paper, an improved BM cells detection algorithm YOLOv7-CTA based on YOLOv7 is proposed. This method can identify BM cells images more accurately than similar models. Experimental analysis shows that a new feature extraction network CoTLAN is designed in the backbone network, which can improve the fine-grained feature extraction capability. The coordinate attention (CoordAtt) module is combined in the network to make the model pay more attention to the features of the area to be detected, reduce irrelevant features, and improve the detection effect of the model. Finally, under the determined network structure, the model is optimized through the selection of loss function, the use of K-means++ algorithm for clustering the target frame of BM cells dataset, and the replacement of cross entropy. The experimental results show that the mAP of the optimized model reaches 88.6%, surpassing the Faster R-CNN, YOLOv5l, and YOLOv7 models by 13.9%, 8.3%, and 6.7%, respectively. Furthermore, the detection speed of this model is 22 fps, effectively satisfying the requirements for high performance. Compared with other models, the YOLOv7-CTA model has superiority in the BM cell detection task.

Implementation of paper - @link

| ![PR_curve.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/runs/test/exp_v7/PR_curve.png?raw=true) | ![PR_curve.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/runs/test/exp_v7-CTA_Focal_CIoU/PR_curve.png?raw=true) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| (**a**)                                                      | (**b**)                                                      |

**Figure** **1**. Precision–recall curve of different models: (a) YOLOv7 model; (b)YOLOv7-CTA.

| ![confusion_matrix.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/runs/test/exp_v7/confusion_matrix.png?raw=true) | ![confusion_matrix.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/runs/test/exp_v7-CTA_Focal_CIoU/confusion_matrix.png?raw=true) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| (**a**)                                                      | (**b**)                                                      |

**Figure** **2**. Confusion matrix of different models: (a) YOLOv7 model; (b)YOLOv7-CTA.

| ![precision.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/precision.png?raw=true) | ![mAP50.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/mAP50.png?raw=true) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| (**a**)                                                      | (**b**)                                                      |

**Figure** **3**. The training results of various models were compared: (a) precision curve; (b) mAP@0.5 curve.

| ![cls_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/cls_loss.png?raw=true) | ![box_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/box_loss.png?raw=true) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| (**a**)                                                      | (**b**)                                                      |

**Figure** **4**. The training loss curves of various models were compared: (a) cls_loss curve; (b) box_loss curve.

![loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/loss.png?raw=true)

**Figure** **5**. The loss curves of the models were compared on the testing dataset. 

Demo

- ![cls_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/1.png?raw=true)

  ​                                  **Figure** **6**. Detection outcomes of images containing fine-grained feature objects and complex backgrounds.

  ![cls_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/2.png?raw=true)

  ​                                                                        **Figure** **7**. Detection results for images with dense objects.

  ![cls_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/3.png?raw=true)

  ​                                                              **Figure** **8**. Detection results for images with objects of complex features. 

## Performance 

BM cells datasets （GPU：NVIDIA GeForce GTX1660 Ti (6G)）

| **Models**   | **Backbone**      | **mAP@0.5/%** | **FPS** |
| ------------ | ----------------- | ------------- | ------- |
| Faster R-CNN | ResNet50          | 74.7          | 7       |
| YOLOv5l      | CSPDarkNet53      | 80.3          | 25      |
| YOLOv7       | CBS + ELAN        | 81.9          | 26      |
| YOLOv7-CTA   | CBS + CoTLAN + CA | 88.6          | 22      |

## Testing

``` bash
python test.py --data data/custom.yaml --img 640 --batch 4 --conf 0.001 --iou 0.65 --device 0 --weights runs/exp_v7-CTA_K-means++_Focal_CIoU/weights/best.pt --name exp_v7-CTA
```

## Training

Single GPU training

``` bash
python train.py --workers 0 --device 0 --batch-size 4 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7_CTA.yaml --weights --name exp_v7-CTA_K-means++_Focal_CIoU --hyp data/hyp.scratch.custom.yaml
```

## Inference

On image:
``` bash
python detect.py --weights runs/exp_v7-CTA_K-means++_Focal_CIoU/weights/best.pt --conf 0.25 --img-size 640 --source data/val/images/
```
