# Deep learning

## Basic

1. LSF

   ```
   J=0; dw1=0; dw2=0; db=0;
   for i = 1 to m
   z(i) = wx(i)+b;
   a(i) = sigmoid(z(i));
   J += -[y(i)log(a(i))+(1-y(i)）log(1-a(i));
   dz(i) = a(i)-y(i);
   dw1 += x1(i)dz(i);
   dw2 += x2(i)dz(i);
   db += dz(i);
   J /= m;
   dw1 /= m;
   dw2 /= m;
   db /= m;
   w=w-alpha*dw
   b=b-alpha*db
                         
                         
   Vectorize implementation
    Z = np.dot(w.T,x)
    A = sigmoid(Z)
    dZ = A-Y
    dw = 1/m*X*(dz.T)
    db = 1/m*np.sum(dz)
    w = w - alpha*dw
    b = b - alpha*db                   
   ```

   ```
   funtion_normalizeRow
   
       x_ = np.linalg.norm(x, axis=1, keepdims = True)
       
       x = x / x_
   
       return x
       
   
   ```
   
   A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b∗∗c∗∗d, a) is to use:
   
   ```python
   X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
   
   ```
   
   
   
   > **What you need to remember:**
   >
   > Common steps for pre-processing a new dataset are:
   >
   > - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
   > - Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
   > - "Standardize" the data
   
2. NN

   ![1557572184845](D:\Userlist\桌面\assets\1557572184845.png)

   ```python
   	dZ2 = A2 - Y
       dW2 = 1/m*dZ2.dot(A1.T)
       db2 = 1/m*np.sum(dZ2,axis = 1, keepdims = True)
       dZ1 = W2.T.dot(dZ2) * (1 - np.power(A1, 2))
       dW1 = dZ1.dot(X.T)
       db1 = 1/m*np.sum(dZ1, axis = 1, keepdims = True)
   ```

3. 多层NN

   ​	**Note** that for every forward function, there is a corresponding backward function. That is why at every step of your forward module you will be storing some values in a cache. 

   几层就有几个激活函数	In deep learning, the "[LINEAR->ACTIVATION]" computation is counted as a single layer in the neural network, not two layers.
   
   ![1557667654873](D:\Userlist\桌面\assets\1557667654873.png)

## Improve

1. #### 一些需要考虑的问题

   > 1. 小数据时代七三分或者六二二分是个不错的选择，在大数据时代各个数据集的比例可能需要变成98%：1%：1%，甚至训练集的比例更大。
   >
   > 2. #### <u>确保验证集和测试集来自于同一分布</u>
   >
   > 3. 测试集的目的是对最终所选定的神经网络系统做出无偏估计，如果不需要无偏估计也可以不设置测试集。
   >
   > 4. 机器学习比较在意bias-variance trade-off,但深度学习的误差很少权衡两者，我们总是分别讨论偏差或者方差，却很少谈及偏差和方差的权衡问题
   >
   > 5. 采用双边误差更逼近导数：使用泰勒定理推导

2. 详细介绍

   > ##### 正则化：旨在消除方差以及偏差的一系列方法：数据扩增（用于修正方差）L2正则化、dropout正则化（用于修正方差）
   >
   > 1. 执行方面的小建议：使用新的J，包含第二个正则化项。
   > 2. dropout本质：每次都训练不同的子集，均摊各个权重，且使用在训练期，前向和后向都使用
   > 3. L2正则化，缺点是必须尝试很多正则化参数λ，early stopping 的优点是只运行一次梯度下降，你可以找出W 的较小值，中间值和较大值，

3. 使用权重初始化解决爆炸或者坍塌

   > 原因：L>I，产生exploding，L<1, 产生vanish
   > $$
   > cost = W^{L-1}X
   > $$
   > 因此初始权重尽可能逼近1，以下为普遍法则
   >
   > 1. 如果是常用relu激活函数，W = np.random.randn(shape)*np.sqrt(2/n^(L-1))
   > 2. 如果是tanh，    np.sqrt(1/n^(L-1)) or  np.sqrt(2/(n^(L-1) + n^L)

4. 梯度检验注意事项

   > 1. 只用在debug
   > 2. 带上正则化
   > 3. 不能与dropout同时使用
   > 4. 在这样一种情况：只有W和b接近0的时候BP才是正确的。在随机初始化过程中，运行梯度检验，然后再训练网络

#### dropout 和L2正则化使用代码

```python
# Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
            
            
               
            
            FORWARD_PROPAGATION
    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
    D1 = np.random.rand(A1.shape[0], A1.shape[1])                                         # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob                                         # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = D1 * A1                                         # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob                                         # Step 4: scale the value of neurons
    
    
    
    		BACKWARD_PROPAGATION
     ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = D2 * dA2              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut dow
```

#### 梯度检验代码

```
To compute J_plus[i]:
    Set θ+ to np.copy(parameters_values)
    Set θ+i to θ+i+ε
    Calculate J+i using to forward_propagation_n(x, y, vector_to_dictionary(θ+ )).
To compute J_minus[i]: do the same thing with θ−
Compute gradapprox[i]=J+i−J−i2ε

    for i in range(num_parameters): 
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y ,vector_to_dictionary(theta_plus))
        ### END CODE HERE ###
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y ,vector_to_dictionary(theta_minus))
        ### END CODE HERE ###
        
        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        ### END CODE HERE ###
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox) 
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
    difference = numerator / denominator
    ### END CODE HERE ###
```

### 优化算法

1. mini-batch

   > 1. 样本集(m<=2000)，使用batch梯度下降
   > 2. mini-batch尝试几个不同的2次方，一般为64到512，这是考虑到CPU/GPU内存相符

2. 指数加权平均（Exponentially weighted averages)

3. Gradient descent with momentum![1557836322066](D:\Userlist\桌面\assets\1557836322066.png)
   $$
   \begin{align}
   v_{dw} = {\beta}v_{dw} \ +\  (1 - \beta)dW\\
   v_{db} = \beta v_{db} \ + \ (1-\beta)db\\
   W = W\ -\ \alpha v_{dv}, \ \ b = b\ - \ \alpha v_{db}
   \end{align}
   $$

4. RMSprop------root mean square prop

   ![1557840557631](D:\Userlist\桌面\assets\1557840557631.png)

5. Adam-----Adaptive Moment Estimation

   ![1557840861113](D:\Userlist\桌面\assets\1557840861113.png)

![1557841107875](D:\Userlist\桌面\assets\1557841107875.png)

> 衰减率![1557841418349](D:\Userlist\桌面\assets\1557841418349.png)
>
> 局部最优问题
>
> 1. 实际上高维空间基本上不存在局部最优（所有都是同一个方向）
> 2. 真正的问题是缓慢走过平稳段，这也是RMSprop、Momentum 和Adam 能够加速学习算法的地方。
> 3. beta1 = 0.9, beat2 = 0.999, epsilon = 10.exp(-8)
> 4. 动量算法需要尽可能大一点的学习率和复杂一点的数据
> 5. 经测试，批次选择16或者8最好

##### 实现代码

1. 梯度

   ```
   (Batch) Gradient Descent:
   X = data_input
   Y = labels
   parameters = initialize_parameters(layers_dims)
   for i in range(0, num_iterations):
       # Forward propagation
       a, caches = forward_propagation(X, parameters)
       # Compute cost.
       cost = compute_cost(a, Y)
       # Backward propagation.
       grads = backward_propagation(a, caches, parameters)
       # Update parameters.
       parameters = update_parameters(parameters, grads)
   Stochastic Gradient Descent:
   X = data_input
   Y = labels
   parameters = initialize_parameters(layers_dims)
   for i in range(0, num_iterations):
       for j in range(0, m):
           # Forward propagation
           a, caches = forward_propagation(X[:,j], parameters)
           # Compute cost
           cost = compute_cost(a, Y[:,j])
           # Backward propagation
           grads = backward_propagation(a, caches, parameters)
           # Update parameters.
           parameters = update_parameters(parameters, grads)
   ```

2. 批量

   ```
   # GRADED FUNCTION: random_mini_batches
   
   def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
       """
       Creates a list of random minibatches from (X, Y)
       
       Arguments:
       X -- input data, of shape (input size, number of examples)
       Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
       mini_batch_size -- size of the mini-batches, integer
       
       Returns:
       mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
       """
       
       np.random.seed(seed)            # To make your "random" minibatches the same as ours
       m = X.shape[1]                  # number of training examples
       mini_batches = []
           
       # Step 1: Shuffle (X, Y)
       permutation = list(np.random.permutation(m))
       shuffled_X = X[:, permutation]
   
       shuffled_Y = Y[:, permutation].reshape((1,m))
   
       # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
       num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
       for k in range(0, num_complete_minibatches):
           ### START CODE HERE ### (approx. 2 lines)
           mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
           mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
           ### END CODE HERE ###
           mini_batch = (mini_batch_X, mini_batch_Y)
           mini_batches.append(mini_batch)
       
       # Handling the end case (last mini-batch < mini_batch_size)
       if m % mini_batch_size != 0:
           ### START CODE HERE ### (approx. 2 lines)
           mini_batch_X = shuffled_X[:, num_compl ete_minibatches * mini_batch_size:]
           mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
           ### END CODE HERE ###
           mini_batch = (mini_batch_X, mini_batch_Y)
           mini_batches.append(mini_batch)
       
       return mini_batches
   ```

### 调试超参数

1. 由粗到细

2. 坐标选取范围先是用对数。然后在精细

3. 超参数训练的方式：1、熊猫方式       2、鱼子酱方式（优先）

4. batch归一化（应用于输入层和隐藏层Z），其b没有意义

   ![1557906735650](E:\local\deep_learning\assets\1557906735650.png)

![1557907302188](E:\local\deep_learning\assets\1557907302188.png)





![1557909925405](E:\local\deep_learning\assets\1557909925405.png)

Tessorflow 基础

> 1. #### create placeholders
>
>    ```
>    X = tf.placeholder(tf.float32, (n_x, None))
>    Y = tf.placeholder(tf.float32, (n_y, None))
>    ```
>
> 2. #### Initializing the parameters
>
>    ```
>        tf.set_random_seed(1)                   # so that your "random" numbers match ours
>            
>        ### START CODE HERE ### (approx. 6 lines of code)
>        W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
>        b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
>    ```
>
> 3. #### Forward propagation in tensorflow
>
>    ```
>        Z1 = tf.matmul(W1,X) + b1                                              # Z1 = np.dot(W1, X) + b1
>        A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
>        Z2 = tf.matmul(W2,A1) + b2                                              # Z2 = np.dot(W2, a1) + b2
>        A2 = tf.nn.relu(Z2)                                          # A2 = relu(Z2)
>        Z3 = tf.matmul(W3,A2) + b3   
>     you don't need a3
>    ```
>
> 4. Compute cost
>
>    ```
>    It is important to know that the "logits" and "labels" inputs of tf.nn.softmax_cross_entropy_with_logits are expected to be of shape (number of examples, num_classes). We have thus transposed Z3 and Y for you.
>    Besides, tf.reduce_mean basically does the summation over the examples.   
>       
>       
>       # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
>            logits = tf.transpose(Z3)
>        labels = tf.transpose(Y)
>        
>        ### START CODE HERE ### (1 line of code)
>        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
>    
>    ```
>
> 5. #### Backward propagation & parameter updates
>
>    ```
>    For instance, for gradient descent the optimizer would be:
>    
>    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
>    To make the optimization you would do:
>    
>    _ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
>    ```
>
> 6. #### Building the model
>
>    ```
>    def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
>              num_epochs = 1500, minibatch_size = 32, print_cost = True):
>    ```
>
>    
>
> 7. #### Test with your own image
>
> 8. #### ```,optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost),To make the optimization you would do:,,_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}),```,,
>
> 
>
> 
>
> 
>
> 
>
> 
>
> 

### Structuring Machine Learning Projects

![1557988118451](E:\local\deep_learning\assets\1557988118451.png)

