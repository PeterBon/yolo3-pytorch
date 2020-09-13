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