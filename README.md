# Info:

Porting StarGAN-Keras to Tensorflow 2.5+

# Updated requirements:
The code was updated and last tested scuessfully with the following updated packages:
```
Tensorflow: 2.5.0
Keras: 2.4.3
OpenCV: 4.5.1
Numpy: 1.20.3 (also tried 1.19.2)
Python: 3.8.5
OS: Ubuntu 20.04 LTS
```

## Note:
The porting right now disables eager_execution() keeping the original code as is. Otherwise, the labels will have to be updated to the new Tensorflow tensors instead of lists. Furthermore, Adam optimizer is used using the tf.compat.v1 method.