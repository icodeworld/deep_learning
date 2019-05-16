# 研究生必要

1.  科学哲学：方法论体系
2. 康托集合论  矩阵方法 离散结构 图论方法   群论方法。
3. 有什么用 哪里用 如何用（学之前）
4. 看透文献动机
5. 具体理论还有很多，很多高深理论无意义，不要陷进去 









# 制作数据集

加速

```python
coor = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	图片和标签的批获取
coord.request_stop()
coord.join(threads)
```





# 卷积CNN

##### 借助卷积核提取特征后，送入全连接网络

```
主要模块：卷积	激活	池化 全连接（FC）
发展历史：Lenet-5 AlexNet VGGNet	googleNet ResNet
```

VALID(不全0填充）	输出图片边长 =（输入图片边长 - 卷积核长+1）/步长（向上取整）

​	padding：SAME 全0填充 	输出图片边长 = 入长/步长（向上取整）

![1557370835642](D:\Userlist\桌面\assets\1557370835642.png)

