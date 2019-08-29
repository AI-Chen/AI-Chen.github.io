---
layout: post
title: Built-in Foreground/Background Prior for Weakly-Supervised Semantic Segmentation
categories: Weakly-Supervised-Semantic-Segmentation
tags: 方法
author: Chen
description: 针对弱监督语义分割的固有前景/背景先验信息
---

**摘要**：利用对象性的先验信息来提高Weakly-Supervised Semantic Segmentation的定位精度需要pixel-level annotations/bounding boxes。为了在仅有图像级标签时提高定位精度。作者提出了一种新方法-从预训练的网络本身提取明显更准确的掩模，放弃外部对象模块。这是通过使用密集CRF平滑的更高级卷积层的激活来实现的。

---

### 1、Weakly-Supervised Semantic Segmentation的研究现状（截止2016）
&emsp;&emsp;弱监督语义分割中的<b>“弱”体现在对标注要求的变弱</b>。从像素级的标注演变为image level、labeled bounding等相对而言不那么费时费力的弱标注。一般来讲，这种弱监督的标注比原始的像素级标注更容易获取。当我们将像素级标注的训练数据转变为labeled bounding甚至是更弱的image level的标注后。那么，一个关键性的问题便出现了：<b>How to build the relationship between image-level labels and pixels</b>。也就是说，如何去构建image-level的标签语义和像素点的关联，推断出图像所对应的segmentation mask，从而利用全卷积神经网络去学习分割模型。<br><br>
<center>![image level label](/assets/images/Build prior/1.jpg) </center>
<center>image level label：只需要把上面的图片打上狗的标签即可</center><br><br>
<center>![labeled bounding](/assets/images/Build prior/2.jpg)</center>
<center>labeled bounding：将对象标上方框以及属于哪一类</center> <br><br>

* Pathak, D., Shelhamer, E., Long, J., Darrell, T.: Fully convolutional multi-class multiple instance learning. In: ICLR Workshop. (2015)
* Papandreou, G., Chen, L.C., Murphy, K.P., Yuille, A.L.: Weakly- and semisupervised learning of a deep convolutional network for semantic image segmentation. In: The IEEE International Conference on Computer Vision (ICCV). (December 2015)
* Pathak, D., Krahenbuhl, P., Darrell, T.: Constrained convolutional neural networks for weakly supervised segmentation. In: The IEEE International Conference on Computer Vision (ICCV). (December 2015)

&emsp;&emsp;Pathak的这篇论文初步解决这个问题的尝试。它是第一种考虑仅在弱监督分割环境中使用图像级标签微调预训练CNN的方法。但是它依赖于简单的多示例损失函数所以精度不高。在这个工作的基础上，Papandreou加入了自适应前景/背景偏差形式的先验信息。这种偏差显着提高了准确性，但是，这种偏差是依赖于数据的，此外，在对象定位方面，结果仍然不准确。与此同时，Pathak在image level标记的基础上引入了额外对象大小尺寸的先验信息。但是，这些方法都不利用关于对象位置的任何信息，因此产生较差的定位精度。为了克服定位不准确的问题，以下三篇文章中提出了对象性的概念（the notion of objectness）：
* Pinheiro, P.O., Collobert, R.: From image-level to pixel-level labeling with convolutional networks. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). (June 2015)
* Bearman, A., Russakovsky, O., Ferrari, V., Fei-Fei, L.: What’s the point: Semantic segmentation with point supervision. ArXiv e-prints (2015)
* Wei, Y., Liang, X., Chen, Y., Jie, Z., Xiao, Y., Zhao, Y., Yan, S.: Learning to segment with image-level annotations. Pattern Recognition (2016)

&emsp;&emsp;Pinheiro利用后处理步骤，使用BING或MCG获得的对象提案来平滑其初始分割结果。虽然它改进了本地化，但作为后处理步骤，此过程无法从初始分割所犯的一些错误中恢复这些错误。Bearman和Wei的文章开始使用比image level更高一个级别的标注labeled bounding。这虽然改善了定位问题，但是获得标注的成本也相应上升了。

---

### 2、Motivation
&emsp;&emsp;为了解决研究现状中所提到的对象定位精度上的不准确性，同时又不引入额外的先验信息或者使用更强的监督（labeled bounding）。作者提出了一种新方法-从预训练的网络本身提取明显更准确的掩模，放弃外部对象模块。这是通过使用密集CRF平滑的更高级卷积层的激活来实现的。

---

### 3、模型详述
&emsp;&emsp;模型采用在ImageNet数据集（for the task of object recognition）上预训练过的VGG16作为编码器。它在一个支路上以VGG16中con4与conv5后的激活作为输入。先分别在通道上做全局平均池化后将它们俩对象像素相加进行融合。融合后的图像送入全连接CRF（条件随机场）中做优化处理。最后上采样到原始图像大小得到一个相对优化的分割图mask。这个分割图在最终构造损失函数的时候至关重要。主干网络上在VGG16进行编码后直接将得到的map上采样到原始图像大小得到score map。最后score map与mask、image level label一起送入Loss function中计算loss以进行方向传播。整体的模型图与编码器详细结构如下：<br><br>
<center>![模型结构](/assets/images/Build prior/3.jpg) </center>
<center>模型结构</center><br><br>
<center>![编码器结构](/assets/images/Build prior/4.jpg)</center>
<center>编码器结构</center> <br><br>
&emsp;&emsp;这里会有几点疑惑：

* 通过conv4与conv5融合得到的是什么样的。它为什么能做掩模？
* loss function是怎么构造的？

&emsp;&emsp;这两个问题在论文中一一得到了回答。

#### 3.1、通过conv4与conv5融合得到的是什么样的？它为什么能做掩模？
&emsp;&emsp;可视化结果显示：VGG网络的前两个卷积层提取图像边缘。随着我们在网络中的深入，卷积层提取更高级别的功能。特别是，第三个卷积层激活表示对象形状。第四层表示完整对象的位置，第五层表示最具辨别力的对象部分。同时，观察这两个激活的融合结果也可以发现为什么要全连接CRF去优化它（直接融合后的图像存在大量噪声）。<br><br>
<center>![可视化激活](/assets/images/Build prior/5.jpg) </center>
<center>可视化结果</center><br><br>
&emsp;&emsp;具体conv4与conv5的融合方式为：首先通过512个通道上的平均池操作将这两个层从3D张量（512×W×H）转换为2D矩阵（W×H）。 然后，我们通过简单的元素和将两个结果矩阵融合，并将得到的值在0和1之间缩放。
&emsp;&emsp;至于全连接CRF，它的处理可以参考deeplab v1。不过这里的全连接CRF应该是可微分的。而可谓分的全连接CRF也早已经有了方法。
&emsp;&emsp;最后，融合结果为什么能做掩模也一目了然（因为它区分了前景和背景）。

#### 3.2、loss function是怎么构造的？
&emsp;&emsp;这里的loss function分两个。一个重点在语义，一个重点在位置（这是这篇文章为什么能提升对象定位精度的关键点）
##### 3.2.1、语义的loss function
<center>![语义的loss function](/assets/images/Build prior/6.jpg) </center>
<br><br>
&emsp;&emsp;其中，L代表当前图中存在的类别。L上面加一横代表当前图中不存在的类别。有这样的划分是因为一张图中可能没有包括所有该数据集中存在的类别。![S](/assets/images/Build prior/8.jpg)，表示整个score map中含有k类对象的概率。由以下公式计算：<br><br>
<center>![计算公式1](/assets/images/Build prior/9.jpg)</center>
<br><br>
&emsp;&emsp;这其中，![S](/assets/images/Build prior/11.jpg)表示（i，j）位置像素属于k类的概率。
&emsp;&emsp;所以很显然，loss function中第一项是对网络预测对了当前图像中出现的对象的奖励。![S](/assets/images/Build prior/8.jpg)越大，loss越小。而第二项则是对当前图像没有出现的对象而网络却预测出了的一种惩罚。![S](/assets/images/Build prior/8.jpg)越大，loss越大。

##### 3.2.1、位置的loss function
<center>![位置的loss function](/assets/images/Build prior/7.jpg) </center>
<center>可视化结果</center><br><br>
其中：<br><br>
<center>![计算公式2](/assets/images/Build prior/12.jpg)</center><br><br>
<center>![计算公式1](/assets/images/Build prior/13.jpg)</center><br><br>
&emsp;&emsp;我们让![S](/assets/images/Build prior/14.jpg)表示mask中位置（i，j）处的值。当当前像素为前景时，它的值为1，反之则为0.同时loss function中|M|表示前景的像素点个数。同样，加一横表示背景的像素点个数。
&emsp;&emsp;很显然，第一项与第二项仍然是对预测正确的奖励；第三项是对预测错误的惩罚。

### 4、总结
&emsp;&emsp;这是我第一次正式阅读弱监督语义分割相关论文。对论文本身的方法先不做评价。接下来想认真的复现以下这一篇论文的代码。
