---
layout: post
title: FULLY CONVOLUTIONAL MULTI-CLASS MULTIPLE
categories: Weakly-Supervised-Semantic-Segmentation
tags: 方法
author: Chen
description: 基于FCN的多类别多示例学习
---

**摘要**：作者提出了一种新的MIL Loss，这是第一次试图用MIL（多示例学习）的思想从图像级
的弱标签标签中学习语义分割模型。

---

### 1、关键点
* 在FCN中加入了MIL的思想，使得MIL可以在网络中做成端到端的表示，这消除了实例化实例标签假设的需要。
* <font color=red><b>提出了一种像素级别的多类别MIL Loss。</b></font>(这是这篇文章的key point)
* 从图像级别的标注获得了语义分割的结果

---

### 2、模型结构简述
<div align=center>
![FCN模型](/assets/images/MILLoss/1.jpg)

&emsp;&emsp;如上所示是FCN的整个模型示意图。至于FCN的原理，这里不加赘述。我们知道，这篇文章使用的数据只有图像级别的标注。所以，由于没有像素级别的标注，无法按照常规的FCN的思想采用交叉熵损失函数计算Loss，直接反卷积到输入图片的大小意义不大。那么作者是怎么解决这个问题的？既然直接反卷积到输入图片大小意义不大的话，作者就直接去掉了FCN中反卷积的部分（即图中红色方框部分），保留了编码器部分（即图中蓝色方框部分）。这样的操作之后，模型经过编码器产生一张heatmap（大小为N*H*W,N为该数据集的class的数目)。<br>
&emsp;&emsp;现在我们得到了heatmap，并且我们还有这张图的图片级别的标签。接下来我们怎么设计Loss成了关键点。

---

### 3、MIL Loss
&emsp;&emsp;我们直接将原文中的两个公式贴在这里：
<div align=center>
![Loss](/assets/images/MILLoss/2.jpg)

&emsp;&emsp;第一个公式做了如下工作：<br>
&emsp;&emsp;从N个具有H*W个像素点的heatmap上分别求出能代表第l（l属于LI。LI代表当前图片label中所具有的类。因为一张图片中可能不包括数据集中的所有类，即LI<=N。作者只选择当前图片标签所表示的类)类的概率最大的像素点的概率。用这一点去代表预测当前图片值包含该类的概率。<br>
&emsp;&emsp;第二个公式做了如下工作：<br>
&emsp;&emsp;很显然，这个公式是当前图片label中表示的类的概率（即第一个公式求出来的值）越高，值越小。这样在反向传播的过程中，就会去使得网络预测当前图片像素的概率时，给予当前图片label中含有的类的heatmap更大的概率值。<br>
&emsp;&emsp;这样训练完毕之后，通过双线性插值到输入图像的大小，得到最后的像素分割结果。

---

### 4、实际效果
&emsp;&emsp;先不看实际的效果图你也能猜到效果不会很好。因为我们在通过FCN学习到的语义信息（图像级别的标记）是不对位置有要求的。所以，分割图的边界形状肯定很糟糕。我们具体来看一下：
<div align=center>
![Loss](/assets/images/MILLoss/3.jpg)

<div align=center>
![Loss](/assets/images/MILLoss/4.jpg)

&emsp;&emsp;结果与猜测一致。对于边界的精度肯定做不到全监督那么好。但是这是第一次尝试，为弱监督语义分割开了一个好头。这是它的意义所在。

---

### 5、总结
* 这是一次好的尝试。让人看到了改进的方向；
* 改进方向一个在于本文的Loss中没有对网络预测出当前图片没有的类别给予惩罚；
* 改进方向另一个在于位置精度还很不理想。如何提升位置进度会是研究的热点。
