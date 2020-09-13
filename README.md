## 交通标志检测

### 1.数据集
* tt100k交通标志数据集 https://cg.cs.tsinghua.edu.cn/traffic-sign/
* 分辨率2048x2048
* 100000张图片，其中10000张图片包含30000个交通标志
![](assets/figure1.jpg)
* 存在交通标志类别不均衡问题，具体如下图，表示每个类别的数量
![](assets/figure2.png)
* 小目标最常见，如下图，表示不同大小的目标数量
![](assets/figure3.png)

### 2.数据增强
* 随机裁剪和缩放
* 随机旋转
* 随机对比度和亮度
* 随机饱和度
* 随机变换色度(HSV空间下(-180, 180))

### 3.针对交通标志检测改进yolov3
* yolov3结构
![](assets/figure4.png)
* 用k均值聚类重新选择anchor box，原来是9个（10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326），
参考项目https://github.com/lars76/kmeans-anchor-boxes
* 采用深度可分离卷积，减小计算量，提高检测速度
* GIoU

### 4.结论和对比
* precision（精确度）和recall（召回率）
  *  TP（True Positives）意思就是被分为了正样本，而且分对了。
  * TN（True Negatives）意思就是被分为了负样本，而且分对了，
  * FP（False Positives）意思就是被分为了正样本，但是分错了（事实上这个样本是负样本）。
  * FN（False Negatives）意思就是被分为了负样本，但是分错了（事实上这个样本是这样本）。
![](assets/figure5.png)
![](assets/figure6.png)

