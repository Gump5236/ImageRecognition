
                                 #基于CNN的海面舰船图像二分类
##本报告选取基于VGG16的网络架构模型，实现对海面舰船数据的二分类任务（船类和非船类），报告具体内容如下。

##1.模型依赖的环境和硬件配置
###1.1 Google Colab （硬件配置可以看到显卡型号是Tesla T4，驱动版本SMI是465.19.01，CUDA版本是11.2，显存是15G）


###1.2目录布置

##2.VGG16的网络架构模型细节
VGG16网络架构共有16层，其中卷积层有13层（分别用conv3-XXX表示），全连接层有3层（分别用FC-XXXX表示）。
    ###2.1 卷积是什么：
卷积过程是基于一个小矩阵，也就是卷积核，按照每层像素矩阵上不断按步长扫过去的，扫到数与卷积核对应位置的数相乘，然后求总和，每扫一次，得到一个值，全部扫完则生成一个新的矩阵。（卷积核大小一般为3×3矩阵）
   ###2.2 Padding是什么：
卷积操作之后维度变少，得到的矩阵比原来矩阵小，这样不好计算，所以需要Padding，在每次卷积操作之前，在原矩阵外边补包一层0，可以只在横向补，或只在纵向补，或者四周都补0，从而使得卷积后输出的图像跟输入图像在尺寸上一致。
   ###2.3 Pooling是什么：
卷积操作后我们提取了很多特征信息，相邻区域有相似特征信息，可以相互替代的，如果全部保留这些特征信息就会有信息冗余，增加了计算难度，这时候池化就相当于降维操作。池化是在一个小矩阵区域内，取该区域的最大值或平均值来代替该区域，该小矩阵的大小可以在搭建网络的时候自己设置。小矩阵也是从左上角扫到右下角。
   ###2.4 Flatten是什么：
Flatten 是指将多维的矩阵拉开，变成一维向量来表示。
   ###2.5 全连接层是什么：
对n-1层和n层而言，n-1层的任意一个节点，都和第n层所有节点有连接。即第n层的每个节点在进行计算的时候，激活函数的输入是n-1层所有节点的加权。
   ###2.6 Dropout是什么：
Dropout是指在网络的训练过程中，按照一定的概率将网络中的神经元丢弃，这样有效防止过拟合。
   ###2.7 激活函数是什么：
####1.隐层层激活函数Relu函数（线性修正单元）取最大值函数，具有的优点：在正数区间解决了梯度下降的问题，计算速度非常快，收敛速度快；存在问题：部分神经元可能不会激活，参数无法更新。

####2.输出激活函数softmax含义在于不再唯一的确定某一个最大值，而是输出每个分类结果的概率值，表示这个类别的可能性；将多分类信息，转化为范围在[0,1]之间和为１的概率分布；

VGG16网络结构图

##3.实验结果

   ###3.1 训练结果（首先， 准备数据集：训练集、验证集和测试集，我是这样实现的：从训练数据集中随机抽取30张样本作为验证集，测试集自己按照需要自定义，我是从训练集中随机抽取30张样本作为测试集，这里需要注意的是：测试集样本不能是训练集样本，否则这样没有测试意义。）

   ###3.2 验证成功率（可以看出收敛效果还是很明显的，验证集的测试精度可达到99%）

   ###3.3 测试结果


##4.实验结果总结和分析
###4.1VGG16的优缺点：
1.优点：简化了神经网络结构。
2.缺点：即训练时间过长，调参难度大。需要的存储容量大，不利于部署。例如存储VGG16权重值文件的大小为500多MB，不利于安装到嵌入式系统中。
###4.2实验建议：
1.分类任务中网络的输入尺寸一般都比较小，所以对于原始数据比较大的时候，要考虑resize步骤损失的有效信息，重新改变输入或者调整网络结构。
2.批量数据训练一定要注意损失函数的收敛问题，建议考虑加载预训练模型。
##注：本次实验我使用的公开包和库主要是Opencv和预训练模型vgg16-397923af.pth
