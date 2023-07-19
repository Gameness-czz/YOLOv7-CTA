import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 列出待获取数据内容的文件位置
    # v5、v8都是csv格式的，v7是txt格式的
    result_dict = {
        'YOLOv51': r'D:\python_project\yolov5\runs\train\exp_v5l\results.txt',
        'YOLOv7': r'D:\python_project\yolov7-CTA\runs\train\exp_v7\results.txt',
        'YOLOv7-CTA': r'D:\python_project\yolov7-CTA\runs\train\exp_v7-CoT_CA\results.txt',
    }

    # 绘制map50
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if modelname == 'Faster R-CNN':
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:  # 按行读取
                    data.append(float(d.strip().split()[0]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                data = np.array(data)  # list转为数组
        else:
            if ext == 'csv':
                data = pd.read_csv(res_path, usecols=[6]).values.ravel()  # 按列直接读取，map50是下标=6的列，并通过.ravel()降为一维数组
            else:  # ext='txt'
                with open(res_path, 'r') as f:
                    datalist = f.readlines()
                    data = []
                    for d in datalist:  # 按行读取
                        data.append(float(d.strip().split()[10]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                    data = np.array(data)  # list转为数组
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("mAP50.png")
    plt.show()

    # 绘制map50-95
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[7]).values.ravel()  # map50-95，解释参照上方
        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[11]))  # map50-95，解释参照上方
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5:0.95')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("mAP50-95.png")
    plt.show()

    # 绘制loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()  # 按列直接读取，map50是下标=6的列，并通过.ravel()降为一维数组
        else:  # ext='txt'
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:  # 按行读取
                    data.append(float(d.strip().split()[5]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                data = np.array(data)  # list转为数组
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("loss.png")
    plt.show()

    # 绘制precision
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()  # 按列直接读取，map50是下标=6的列，并通过.ravel()降为一维数组
        else:  # ext='txt'
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:  # 按行读取
                    data.append(float(d.strip().split()[8]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                data = np.array(data)  # list转为数组
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('precision')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("precision.png")
    plt.show()

    # 绘制cls_loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()  # 按列直接读取，map50是下标=6的列，并通过.ravel()降为一维数组
        else:  # ext='txt'
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:  # 按行读取
                    data.append(float(d.strip().split()[9]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                data = np.array(data)  # list转为数组
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('recall')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("recall.png")
    plt.show()

    # 绘制box_loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()  # 按列直接读取，map50是下标=6的列，并通过.ravel()降为一维数组
        else:  # ext='txt'
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:  # 按行读取
                    data.append(float(d.strip().split()[2]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                data = np.array(data)  # list转为数组
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('box_loss')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("box_loss.png")
    plt.show()

    # 绘制obj_loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()  # 按列直接读取，map50是下标=6的列，并通过.ravel()降为一维数组
        else:  # ext='txt'
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:  # 按行读取
                    data.append(float(d.strip().split()[3]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                data = np.array(data)  # list转为数组
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('obj_loss')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("obj_loss.png")
    plt.show()

    # 绘制cls_loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()  # 按列直接读取，map50是下标=6的列，并通过.ravel()降为一维数组
        else:  # ext='txt'
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:  # 按行读取
                    data.append(float(d.strip().split()[4]))  # 将该行下标=10（该轮训练得到的map50）的位置插入到data列表中
                data = np.array(data)  # list转为数组
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('cls_loss')
    plt.legend()
    plt.grid()
    # 保存并显示图像
    plt.savefig("cls_loss.png")
    plt.show()