---
layout: post
title:  Efficient Ladder style DenseNets总结
---

**摘要**：GPU内存的大小是影响语义分割效率的一个重要因素。目前，正常内存大小的GPU，在对大分辨率图像大小进行分割时，如何支持卷积反向传播所需的特征图的高速缓存是一个重大挑战（即使对于中等分辨率大小的Pascal图像也是如此）。DenseNet被提了出来用于解决这一问题。但是由于目前主流的深度学习编程框架没有很好的算法支持，导致DenseNet在进行语义分割时所需要的内存以及分割的效率与ResNet差别不大。为了解决这一问题，作者基于DenseNets精细的设计了Efficient Ladder style DenseNets模型。在该模型中：1）对 DenseNet 进行了详尽的研究；2）提出阶梯式上采样路径，将深层特征的语义与早期层的位置精度相结合，并且不需要大量的计算资源；3）支持在反向传播期间积极地重新计算中间激活，通过Gradient checkpointing减少与反向传播相关的缓存的大小。最终的模型，在Citycapes、CamVid、Pascal VOC 2012等数据集上取得了优秀的分割结果。

----
### 1、DenseNet
#### 1.1、Dense block
&emsp;&emsp;DenseNet 是 CVPR 2017 最佳论文之一。最近一两年卷积神经网络提高效果的方向，一方面是使得网络的深度加深（比如 ResNET，通过残差块解决了网络加深时候的梯度消失问题从而使得网络得以发展到数百层之深），另一方面试扩宽网络（比如 GoogleNet 的 Inception模块），而DenseNet的作者则是从特征入手，通过对特征的极致利用达到更好的效果和更少的参数。<br>
&emsp;&emsp;先放一个 DenseNet block 的基本构造，对其有一个整体的感知。如下图：在该网络中，任何两层之间都有直接的连接，也就是说，网络每一层的输入都是前面所有层输出的并集，而该层所学习的特征图也会被直接传给其后面所有层作为输入。<br><br>
![DenseNet框架](../assets/images/Ladder style DenseNets/LDN1.JPG)<br><br>
&emsp;&emsp;如果记第 l 层的变换函数为Hi（通常对应于一组或两组3*3 卷积、批归一化、ReLU
操作），输出为 xi，那么每一层的 DenseNet 的操作可以由如下方式表示：
<center>x𝑖 = 𝐻𝑖 (x0, x1, … , x𝑖−1 )</center><br>
&emsp;&emsp;这个方式十分容易理解以及实现。
#### 1.1、Dense block
&emsp;&emsp;将3个Dense block搭建在一起便形成了DenseNet。下面是 DenseNet 的整个框架图以及具体的实现细节：<br><br>
![DenseNet框架](../assets/images/Ladder style DenseNets/LDN2.JPG)<br><br>
![DenseNet框架](../assets/images/Ladder style DenseNets/LDN3.JPG)<br><br>
> * 每一个Dense block中的卷积操作最后产生的特征图的通道数(channel)是一个固定值，称为增长率，并由超参数k定义。
* Dense Block 结束后的输出 channel个数很多，需要用1×1 的卷积核来降维。Transition Layer 中经 1×1 卷积后输出的通道数与输入之比称为压缩率（theta）。
* bottleneck与transition layer：在每个 Dense Block 中都包含很多个子结构，以 DenseNet-169 的 Dense Block（3）为例，它包含 32 个 1×1 （在3×3卷积前加上1×1卷积即为bottleneck）和 3×3 的卷积操作，也就是第 32 个子结构的输入是前面 31 层的输出结果，每层输出的 channel 是 32（增长率k），那么如果不做bottleneck 操作，第 32 层的 3×3 卷积操作的输入就是 3×3+（上一个 Dense Block的输出 channel），近 1000 了。而加上 1×1  的卷积，经过 1×1 卷积后的 channel是 growth rate×4，也就是 128，然后再作为 3×3 卷积的输入。这就大大减少了计算量，这就是bottleneck。至于 transition layer，放在两个 Dense Block 中间，是因为每个 Dense Block 结束后的输出 channel 个数很多，需要用 1×1 的卷积核来降维。还是以 DenseNet-169 的 Dense Block（3）为例，虽然第 32 层的 3*3 卷积输出 channel 只有 32 个（增长率k），但是紧接着还会像前面几层一样有通道的 concat 操作，即将第 32 层的输出和第 32 层的输入做 concat，前面说过第 32层的输入是 1000 左右的 channel，所以最后每个 Dense Block 的输出也是 1000 多的 channel。因此这个 transition layer 有个参数 压缩率（范围是 0 到 1），表示将这些输出缩小到原来的多少倍，默认是 0.5，这样传给下一个 Dense Block 的时候 channel 数量就会减少一半，这就是 transition layer 的作用。
* 另外，包含 bottleneck layer（dense block 的 3*3 卷积前面都包含了一个 1*1
的卷积操作，就是所谓的bottleneck layer）的叫DenseNet-B，包含压缩层（transition
layer）的叫 DenseNet-C，两者都包含的叫 DenseNet-BC。

&emsp;&emsp;DenseNet 的想法很大程度上源于在 ECCV 上的一个叫做随机深度网（Deep
networks with stochastic depth）工作。当时提出了一种类似于 Dropout 的方法来改进 ResNet。我们发现在训练过程中的每一步都随机地「扔掉」（drop）一些层，可以显著的提高 ResNet 的泛化性能。这个方法的成功至少带给我们两点启发：<br>
&emsp;&emsp;首先，它说明了神经网络其实并不一定要是一个递进层级结构，也就是说网络中的某一层可以不仅仅依赖于紧邻的上一层的特征，而可以依赖于更前面层学习的特征。想像一下在随机深度网络中，当第 l 层被扔掉之后，第 l+1 层就被直接连到了第 l-1 层；当第 2 到了第 l 层都被扔掉之后，第 l+1 层就直接用到了第 1 层的特征。因此，随机深度网络其实可以看成一个具有随机密集连接的DenseNet。<br>
&emsp;&emsp;其次，我们在训练的过程中随机扔掉很多层也不会破坏算法的收敛，说明了ResNet 具有比较明显的冗余性，网络中的每一层都只提取了很少的特征（即所谓的残差）。实际上，我们将训练好的 ResNet 随机的去掉几层，对网络的预测结果也不会产生太大的影响。<br>
&emsp;&emsp;DenseNet 的设计正是基于以上两点观察。我们让网络中的每一层都直接与
其前面层相连，实现特征的重复利用；同时把网络的每一层设计得特别「窄」，即只学习非常少的特征图（最极端情况就是每一层只学习一个特征图），达到降低冗余性的目的。这两点也是 DenseNet 与其他网络最主要的不同。需要强调的是，第一点是第二点的前提，没有密集连接，我们是不可能把网络设计得太窄的，否则训练会出现欠拟合（under-fitting）现象，即使 ResNet 也是如此。

----
### 2、Efficient Ladder style DenseNets方法概述
#### 2.1、Efficient Ladder style DenseNets的提出动机

#### 2.2、Efficient Ladder style DenseNets整体框架

#### 2.3、阶梯式上采样路径

#### 2.4、Gradient checkpointing（梯度检查点）

----
### 2、实验

----
### 2、总结
