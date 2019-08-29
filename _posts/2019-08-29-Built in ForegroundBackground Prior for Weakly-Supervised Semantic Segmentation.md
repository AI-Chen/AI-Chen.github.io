---
layout: post
title: Built-in Foreground/Background Prior for Weakly-Supervised Semantic Segmentation
categories: Weakly-Supervised Semantic Segmentation
tags: 方法
author: Chen
description: 针对弱监督语义分割的固有前景/背景先验信息
---

**摘要**：利用对象性的先验信息来提高Weakly-Supervised Semantic Segmentation的定位精度需要pixel-level annotations/bounding boxes。为了在仅有图像级标签时提高定位精度。作者提出了一种新方法-从预训练的网络本身提取明显更准确的掩模，放弃外部对象模块。这是通过使用密集CRF平滑的更高级卷积层的激活来实现的。

---

### 1、Weakly-Supervised Semantic Segmentation的研究现状（截止2016）
&emsp;&emsp;弱监督语义分割中的<b>“弱”体现在对标注要求的变弱</b>。从像素级的标注演变为image level、labeled bounding等相对而言不那么费时费力的弱标注。一般来讲，这种弱监督的标注比原始的像素级标注更容易获取。当我们将像素级标注的训练数据转变为labeled bounding甚至是更弱的image level的标注后。那么，一个关键性的问题便出现了：<b>How to build the relationship between image-level labels and pixels</b>。也就是说，如何去构建image-level的标签语义和像素点的关联，推断出图像所对应的segmentation mask，从而利用全卷积神经网络去学习分割模型。<br><br>
![image level label](/assets/images/Build prior/1.jpg) <br><br>
<center>image level label：只需要把上面的图片打上狗的标签即可</center>
![labeled bounding](/assets/images/Build prior/2.jpg) <br><br>
<center>labeled bounding：将对象标上方框以及属于哪一类</center>
&emsp;&emsp;Pathak, D., Shelhamer, E., Long, J., Darrell, T.: Fully convolutional multi-class multiple instance learning. In: ICLR Workshop. (2015)这篇论文初步解决这个问题的尝试。它是第一种考虑仅在弱监督分割环境中使用图像级标签微调预训练CNN的方法。但是它依赖于简单的多示例损失函数所以精度不高。在这个工作的基础上，Papandreou, G., Chen, L.C., Murphy, K.P., Yuille, A.L.: Weakly- and semisupervised learning of a deep convolutional network for semantic image segmentation. In: The IEEE International Conference on Computer Vision (ICCV). (December 2015)加入了自适应前景/背景偏差形式的先验信息，。这种偏差显着提高了准确性，但是，这种偏差是依赖于数据的，此外，在对象定位方面，结果仍然不准确。与此同时，Pathak, D., Krahenbuhl, P., Darrell, T.: Constrained convolutional neural networks for weakly supervised segmentation. In: The IEEE International Conference on Computer Vision (ICCV). (December 2015)在image level标记的基础上引入了额外对象大小尺寸的先验信息。但是，这些方法都不利用关于对象位置的任何信息，因此产生较差的定位精度。为了克服定位不准确的问题，Pinheiro, P.O., Collobert, R.: From image-level to pixel-level labeling with convolutional networks. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). (June 2015)、Bearman, A., Russakovsky, O., Ferrari, V., Fei-Fei, L.: What’s the point: Semantic segmentation with point supervision. ArXiv e-prints (2015)、Wei, Y., Liang, X., Chen, Y., Jie, Z., Xiao, Y., Zhao, Y., Yan, S.: Learning to segment with image-level annotations. Pattern Recognition (2016)这三篇文章中提出了对象性的概念（the notion of objectness）。Pinheiro利用后处理步骤，使用BING或MCG获得的对象提案来平滑其初始分割结果。虽然它改进了本地化，但作为后处理步骤，此过程无法从初始分割所犯的一些错误中恢复。Bearman和Wei的文章开始使用比image level更高一个级别的标注labeled bounding。这虽然改善了定位问题，但是获得标注的成本也相应上升了。

---

### 2、Motivation
&emsp;&emsp;为了解决研究现状中所提到的对象定位精度上的不准确性，同时又不引入额外的先验信息或者使用更强的监督（labeled bounding）。作者提出了一种新方法-从预训练的网络本身提取明显更准确的掩模，放弃外部对象模块。这是通过使用密集CRF平滑的更高级卷积层的激活来实现的。

---

### 3、模型详述
&emsp;&emsp;
