# Official YOLOv7-CTA

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

**Figure ** **5**. The loss curves of the models were compared on the testing dataset.

Demo

- ![cls_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/1.png?raw=true)

  ​                                  **Figure** **6**. Detection outcomes of images containing fine-grained feature objects and complex backgrounds.

  ![cls_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/2.png?raw=true)

  ​                                                                        **Figure** **7**. Detection results for images with dense objects.

  ![cls_loss.png](https://github.com/Gameness-czz/YOLOv7-CTA/blob/main/contrast_results/3.png?raw=true)

  ​                                                              **Figure** **8**. Detection results for images with objects of complex features. 

## Performance 

BM cells datasets

| **Models**   | **Backbone**      | **mAP@0.5/%** | **FPS** |
| ------------ | ----------------- | ------------- | ------- |
| Faster R-CNN | ResNet50          | 74.7          | 7       |
| YOLOv5l      | CSPDarkNet53      | 80.3          | 25      |
| YOLOv7       | CBS + ELAN        | 81.9          | 26      |
| YOLOv7-CTA   | CBS + CoTLAN + CA | 88.6          | 22      |

## Testing

``` bash
python test.py --data data/custom.yaml --img 640 --batch 4 --conf 0.001 --iou 0.65 --device 0 --weights '' --name exp_v7-CTA
```

## Training

Single GPU training

``` bash
python train.py --workers 0 --device 0 --batch-size 4 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7_CTA.yaml --weights '' --name exp_v7-CTA --hyp data/hyp.scratch.custom.yaml
```

## Inference

On image:
``` bash
python detect.py --weights runs/exp_v7-CTA --conf 0.25 --img-size 640 --source data/val/images/
```
