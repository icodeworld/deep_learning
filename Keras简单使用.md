[TOC]

# Keras简单使用

####  在keras中建立模型

相对于自己写机器学习相关的函数，keras更能快速搭建模型，流程如下：

1. 通过调用下面的函数创建模型
2. 通过调用 `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`编译模型
3. 通过调用 `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`在训练集上训练模型
4. 通过调用`model.evaluate(x = ..., y = ...)`在测试集上测试模型

如果你想查阅更多有关`model.compile()`, `model.fit()`, `model.evaluate()` 的信息和它们的参数, 请参考官方文档 [Keras documentation](https://keras.io/models/model/).

代码如下：

```python
def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model
```

```python
step 1:
happyModel = HappyModel(X_train.shape[1:]) # 只保留一个例子

step 2:
happyModel.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

step 3：
happyModel.fit(x = X_train,y = Y_train, epochs = 5, batch_size = 16)

step 4：
preds =  happyModel.evaluate(x = X_test, y = Y_test)
# preds[0] = Loss
# preds[1] = Test Accuracy
```

此时，模型参数均已确定，可用来测试自己的图片

#### 测试自己的图片

```python
img_path = 'your picture path'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))
```

#### 一些有用的函数(持续更新)

1. `happyModel.summary()`：统计并打印如下内容

   ![img](https://img2018.cnblogs.com/blog/1692206/201905/1692206-20190519191516373-97446917.png)

2. `plot_model()`画出流程图

   ```python
   plot_model(happyModel, to_file='HappyModel.png')
   SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
   ```

   ![img](https://img2018.cnblogs.com/blog/1692206/201905/1692206-20190519191011741-1270005326.png)

参考资料：[https://github.com/Kulbear/deep-learning-coursera/blob/master/Convolutional%20Neural%20Networks/Keras%20-%20Tutorial%20-%20Happy%20House%20v1.ipynb](https://github.com/Kulbear/deep-learning-coursera/blob/master/Convolutional Neural Networks/Keras - Tutorial - Happy House v1.ipynb)