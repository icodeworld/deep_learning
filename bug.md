日常坑：

##### 2019-5

1. `db = np.sum(dZ, axis=1, keepdims=True) / m`must write ‘axis = 1’， otherwise ，后面会出现错误。

2. np.random.rand(A1.shape) 会报错，解决方法 

   ```
   1，np.random.rand(A1.shape[0], A1.shape[1])
   2. np.random.rand(*A1.shape)
   ```

3. python若最后一个元素直接为空即可如 `A[2:]`，`A[2:-1]`表示不包含最后一个元素

4. numpy.sqrt()

5. ppps:如果想在任意文件夹下打开notebook,可以打开conda prompt，激活(activate)你需要用到的虚拟环境，然后cd进入你的程序项目所在的文件夹，直接输入jupyter notebook就可以啦。记得把kernel换成相对应的虚拟环境所生成的内核。