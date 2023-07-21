import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def bboxesOverRation(bboxesA, bboxesB):
    """
    功能等同于matlab的函数bboxesOverRation
    bboxesA：M*4 array,形如[x,y,w,h]排布
    bboxesB: N*4 array,形如[x,y,w,h]排布
    """
    bboxesA = np.array(bboxesA.astype('float'))
    bboxesB = np.array(bboxesB.astype('float'))
    M = bboxesA.shape[0]
    N = bboxesB.shape[0]

    areasA = bboxesA[:, 2] * bboxesA[:, 3]
    areasB = bboxesB[:, 2] * bboxesB[:, 3]

    xA = bboxesA[:, 0] + bboxesA[:, 2]
    yA = bboxesA[:, 1] + bboxesA[:, 3]
    xyA = np.stack([xA, yA]).transpose()
    xyxyA = np.concatenate((bboxesA[:, :2], xyA), axis=1)

    xB = bboxesB[:, 0] + bboxesB[:, 2]
    yB = bboxesB[:, 1] + bboxesB[:, 3]
    xyB = np.stack([xB, yB]).transpose()
    xyxyB = np.concatenate((bboxesB[:, :2], xyB), axis=1)

    iouRatio = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x1 = max(xyxyA[i, 0], xyxyB[j, 0]);
            x2 = min(xyxyA[i, 2], xyxyB[j, 2]);
            y1 = max(xyxyA[i, 1], xyxyB[j, 1]);
            y2 = min(xyxyA[i, 3], xyxyB[j, 3]);
            Intersection = max(0, (x2 - x1)) * max(0, (y2 - y1));
            Union = areasA[i] + areasB[j] - Intersection;
            iouRatio[i, j] = Intersection / Union;
    return iouRatio


def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in tqdm(glob.glob('{}/*xml'.format(path))):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height <= 0 or width <= 0:
            continue

        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            x = xmin + 0.5 * (xmax - xmin)
            y = ymin + 0.5 * (ymax - ymin)
            data.append([x, y, xmax - xmin, ymax - ymin])
    return np.array(data)


def estimateAnchorBoxes(trainingData, numAnchors=9):
    '''
    功能：kmeans++算法估计anchor，类似于matlab函数estimateAnchorBoxes,当trainingData
    数据量较大时候，自写的kmeans迭代循环效率较低，matlab的estimateAnchorBoxes得出
    anchors较快，但meanIOU较低，然后乘以实际box的ratio即可。此算法由于优化是局部，易陷入局部最优解，结果不一致属正常
    cuixingxing150@gmail.com
    Example:
        import scipy.io as scipo
        data = scipo.loadmat(r'D:\Matlab_files\trainingData.mat')
        trainingData = data['temp']

        meanIoUList = []
        for numAnchor in np.arange(1,16):
            anchorBoxes,meanIoU = estimateAnchorBoxes(trainingData,numAnchors=numAnchor)
            meanIoUList.append(meanIoU)
        plt.plot(np.arange(1,16),meanIoUList,'ro-')
        plt.ylabel("Mean IoU")
        plt.xlabel("Number of Anchors")
        plt.title("Number of Anchors vs. Mean IoU")

    Parameters
    ----------
    trainingData : numpy 类型
        形如[x,y,w,h]排布，M*4大小二维矩阵
    numAnchors : int, optional
        估计的anchors数量. The default is 9.

    Returns
    -------
    anchorBoxes : numpy类型
        形如[w,h]排布，N*2大小矩阵.
    meanIoU : scalar 标量
        DESCRIPTION.

    '''

    numsObver = trainingData.shape[0]
    xyArray = np.zeros((numsObver, 2))
    trainingData[:, 0:2] = xyArray
    assert (numsObver >= numAnchors)

    # kmeans++
    # init
    centroids = []  # 初始化中心，kmeans++
    centroid_index = np.random.choice(numsObver, 1)
    centroids.append(trainingData[centroid_index])
    while len(centroids) < numAnchors:
        minDistList = []
        for box in trainingData:
            box = box.reshape((-1, 4))
            minDist = 1
            for centroid in centroids:
                centroid = centroid.reshape((-1, 4))
                ratio = (1 - bboxesOverRation(box, centroid)).item()
                if ratio < minDist:
                    minDist = ratio
            minDistList.append(minDist)

        sumDist = np.sum(minDistList)
        prob = minDistList / sumDist
        idx = np.random.choice(numsObver, 1, replace=True, p=prob)
        centroids.append(trainingData[idx])

    # kmeans 迭代聚类
    maxIterTimes = 100
    iter_times = 0
    while True:
        minDistList = []
        minDistList_ind = []
        for box in trainingData:
            box = box.reshape((-1, 4))
            minDist = 1
            box_belong = 0
            for i, centroid in enumerate(centroids):
                centroid = centroid.reshape((-1, 4))
                ratio = (1 - bboxesOverRation(box, centroid)).item()
                if ratio < minDist:
                    minDist = ratio
                    box_belong = i
            minDistList.append(minDist)
            minDistList_ind.append(box_belong)
        centroids_avg = []
        for _ in range(numAnchors):
            centroids_avg.append([])
        for i, anchor_id in enumerate(minDistList_ind):
            centroids_avg[anchor_id].append(trainingData[i])
        err = 0
        for i in range(numAnchors):
            if len(centroids_avg[i]):
                temp = np.mean(centroids_avg[i], axis=0)
                err += np.sqrt(np.sum(np.power(temp - centroids[i], 2)))
                centroids[i] = np.mean(centroids_avg[i], axis=0)
        iter_times += 1
        if iter_times > maxIterTimes or err == 0:
            break

    anchorBoxes = np.array([x[2:] for x in centroids])
    meanIoU = 1 - np.mean(minDistList)
    print('acc:{:.2f}%'.format(avg_iou(trainingData[:, 2:], anchorBoxes) * 100))
    return anchorBoxes, meanIoU


if __name__ == "__main__":
    np.random.seed(0)
    #  载入数据集，可以使用VOC的xml
    path = 'D:/python_project/Medical_stem_cell/VOC_marrow_cells/Annotations'
    # 生成的anchors的txt文件保存路径
    anchorsPath = 'yolo_anchors++.txt'
    # 生成的anchors数量
    anchors_num = 9
    # 输入的图片尺寸
    input_shape = [640, 640]
    print('Load xmls.')
    data = load_data(path)
    print('Load xmls done.')

    #   使用k聚类算法
    print('K-means++ boxes.')
    anchors, _= estimateAnchorBoxes(data, numAnchors=anchors_num)
    print('K-means boxes done.')
    anchors = anchors *  np.array([input_shape[1], input_shape[0]])
    # 排序
    cluster = anchors[np.argsort(anchors[:, 0])]
    print("聚类结果")
    print(cluster)

    # 保存结果 生成yolo_anchors++.txt文件
    f = open(anchorsPath, 'w')
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()


