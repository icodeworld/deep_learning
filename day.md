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





# Retrieve dimensions from tensor
    m, n_H, n_W, n_C = a_G.get_shape().as_list()