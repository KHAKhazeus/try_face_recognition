CFR

ICCV CVPR ACMMM

ECCV

ICIP ICME ICPR

TPAMI TIP CSVT TMM IJCV PR IVC

NIPS ICML

Email: maozhendong@iie.ac.cn



## 基本知识介绍

[TOC]

### Pattern

- 可观察性

- 可区分性

- 相似性

  模式识别： 区分是否相同或相似

  KDD？

  Adaboost



## 图像识别

划分为同一类别：t-SNE 

### Semantic Gap

Low-level features vs high-level concepts

### 基本框架

```mermaid
graph LR
测量空间 --特征表示--> 特征空间 
特征空间 --特征匹配--> 类别空间
```

特征表示：

```
传统方法：设计特征
深度学习：学习特征
```

特征匹配:

k-means 、SVM、欧氏距离

International conference on ...

表示学习



## 早期图像识别技术

Metric learning?

相似度计算：索引技术

```mermaid
graph LR
特征提取 --> 索引技术
索引技术 --> 相关反馈
相关反馈 --> 重排序
```



### 全局特征提取

- 颜色提取：颜色直方图
- 形状：傅里叶变换、过滤器  高频分量？

- 纹理：LBP

子空间嵌入

特征向量



### 特征变换

提高特征表示性能

manifold learning/embedding: PCA、MDS、ISOMAP、LLE、Laplacian Eigenmap

- 中心化、归一化
- 去相关（线性相关）、白化

深度学习：hash学习，梯度变化，收敛快，维度均衡

信息论：熵



### 索引技术

穷举搜索

改进方式

常用：KD-Tree, LSH(Locality Sensitive Hashing)

```
维度大于一定数，所有索引技术不能少于O(n)
原因： 数据稀疏
```

牺牲最近邻，找(1+$\epsilon$)r即可，且$\epsilon$有保证

二进制哈希：降维之后再量化

SpH球哈希、PTH

Mao ZhenDong ACMMM PTH DBIH



### 相关反馈

再排序

重排序 再做visual information进行统计， 根据网页中的文本再进行排序



### 问题

早期识别关注全局特征，忽略了图像的细节



## 中期图片识别技术

### 词袋模型 03

Bag-of-Words

Bag-of-Visual Words

ImageNet



### 局部特征

图像区块(patch)的向量



#### feature detector特征检测子

- 稳定、重复

```
尖锐点、角点
Harris DoG SURF Harris-Affine Hessian-Affine MSER
```



#### feature descriptor特征描述子

- 判别力，robustness

```
解决亮度问题：梯度统计
SIFT PCA-SIFT GLOH Shape Context ORB COGE
```



视觉词典生成

描述子提取向量之后

对特征空间做特征聚类(K-means, Affinity Propagation)

视觉关键词

Video Google|Vocabulary Tree

视觉关键词直方图



### 倒排索引

Tf-IDF加权



### 查询扩展

先查询之后反过来辅助增多特征点

局部几何验证、乘积量化



## 人脸识别

频域特性：低频-总体，高频-细节

面部区域重要性

性别与年龄



负片+下方光源

识别方法：

- 红外,3D,可见光



### 可见光的识别

I = f(F;L;C)

Lambert漫反射模型（视点无关）



无法找到稳定不变的通用的特征

Appearance-base learning



亮度变化(shading),阴影反应3D形状

Albedo，表面反射率



建模：

- 基于几何特征：

  v = (x1, x2, ..., xn)

  比较距离

  UMD USC????



FERET测试

人脸库



基于规则

基于模板

基于不变特征、外观学习



## 基于肤色特征的检测

颜色空间：RGB，HSV，YIQ，YES，CIE XYZ, CIE LUV

肤色模型： Cr, Cb 定义肤色区域

RGB分布阈值 Multi-model tracking of faces for video communications

肤色呈高斯分布 Detecting human faces in color images  用均值和方差确定阈值

加上不同分布高斯混合模型 Parameterized structure from motion fro 3D adaptive feedback tracking of faces



## 基于AdaBoost的人脸检测

ANN

SVM

Naive Bayes Classifier

AdaBoost

Haar-like矩形特征

分类器设计

Cascade分级分类器技术



### 矩形特征

纹理特征有时候也是

矩形特征值 Haar-like特征

r(x,y,w,h,style)



利用积分图

f(x,y)先积分之后再减



### AdaBoost分类器

弱分类器
$$
h_j(x) = 1 \quad if \quad f_j(x) < \theta_j
$$
弱学习算法寻找阈值 $\theta$
$$
\epsilon_j = sigma w_{t,i}|h_j(x_i) - y_i|
$$
分别计算弱分类器的误差，若表现好则下降权重，剩下的再投入训练 $\beta_i = \frac{\epsilon_i}{1-\epsilon_i}​$

最后得到强分类器，根据错误率投票

$\alpha_i = log(\frac{1}{\beta_i})$ 



遍历不同位置、大小



分级做

第一个特征，召回率高，滤除都是人脸 100%detection rate 50% false positive