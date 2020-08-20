PNN模型解读
==

# 背景

> 在推荐系统、计算广告领域中，通常需要去预测用户行为(是否点击/是否转化)。真实业务场景数据集通常会有很多类别型特征(例如性别、ID号等)，这类特征在经过one-hot后会带来过于稀疏的特征空间。浅层模型例如FM虽然可学习二阶特征交叉但是表达能力有限；而深层模型例如DNN虽然可以学习高阶信息，但是DNN本身并不具备学习特征交叉的能力(不同的field之间并无”且“的运算)，而且过于稀疏的输入也不利于网络学习。
>
> Embedding+MLP结构是DNN应用在CTR预估的标准模式。通常，NN层之间都使用“add operation” ，通过激活函数来引入非线性。作者认为，单纯的“add”也许不足以捕获不同的Filed特征间的相关性，一些相关研究表明“product”相比“add”能更好得捕捉特征间的dependence，因此作者希望**在NN中显示地引入“product”操作**，从而更好地学习不同Field特征间的相关性。

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/1.jpg)

> 所以本文提出了**PNN**这个模型，在embedding层后设计了Product Layer，以显示捕捉**基于Field的二阶特征相关性。**其中的**embedding层**学习种类特征的分布式表示，**product层**捕获种类特征之间的交互特征（学习filed之间的交互特征），**全连接层**捕获高阶交互特征。

# 模型

> PNN模型结构如图所示：

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/2.jpg)

> 各层结构定义如下：

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/3.jpg)

> PNN与标准的「Embedding+MLP」差异仅在于引入了Product Layer，Product Layer左边Z部分是将Embedding层直接原封不动地搬来，右边P部分才是优化的重点。注意，product layer 中每个节点（见蓝色节点）是**两两Field的embedding对应**的“product”结果，而非所有Field的。product 函数的不同选择，PNN也有不同实现，文中尝试了相对常见的向量内积（inner product）和外积（outer product），对应 IPNN 和OPNN。

## IPNN

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/4.jpg)

## OPNN

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/5.jpg)

参考：https://zhuanlan.zhihu.com/p/56651241

# 实验结果

## 实验设置

**数据集** 论文使用2个真实世界的开源数据集，具体如下

1. **Criteo** 含1TB的点击日志，使用连续7天的数据训练，紧接着的下1天作测试。经过negative dawn-sampling和特征映射后，最终包含79.38 M 样本及 1.64M 维特征。
2. **iPinyou** 包含超过10天的点击日志，共 19.5M 样本，经过one-hot后特征共 937.67 K维。沿用该数据集原始的train/test划分，即每个advertiser的最后3天数据作test，其余作train。

**对比方法** 使用 logistic loss，论文对比了LR、FM、CCPM、FNN、IPNN、OPNN以及PNN*，PNN*表示同时加入内积和外积。

1. FM和NN模型的embedding维度设为10。
2. 为防止拟合，LR和FM使用L2正则，NN类模型使用rate=0.5的Dropout。
3. CCPM——1嵌入层+2卷积层 (max pooling) +1隐层；FNN——1嵌入层+3隐层；PNN——1嵌入层+1 product层+3隐层。

## 实验结果

> 不同数据集和指标上的结实验果如下表所示，PNN类模型性能最优

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/6.jpg)

> 论文也做了其他补充实验。在iPinYou数据集上，各模型在不同迭代轮数下的Auc曲线如图2所示，可见PNN模型的收敛速度在iPinYou上也优于其他算法。使用不同的隐层depth和不同类型激活函数，实验结果分别如图所示。

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/7.jpg)

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/8.jpg)

![image](https://github.com/ShaoQiBNU/PNN/blob/master/img/9.jpg)

# 结论

1. PNN的动机很直观，通过在NN的嵌入层和隐层之间引入product层，显示地引入基于field的“product”，从而加强单纯基于“add”的NN的特征相关性学习能力。
2. PNN对所有特征进行无差别化的交叉，一定程度上忽略了原始特征向量中包含的信息，存在局限性。

# 代码

https://github.com/shenweichen/DeepCTR

