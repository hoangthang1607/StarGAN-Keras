<p align="center"><img width="40%" src="./assests/logo.jpg" /></p>

--------------------------------------------------------------------------------

## Usage
### Downloading the dataset
```bash
$ bash download.sh celeba
```

### Train
* python main.py --mode train --batch_size <4,8,16... depends on GPU memory>

### Test on celebA images
* python main.py --mode test 

### Test on custom images (please put your images into './samples' directory)
* python main.py --mode custom --custom_image_name <000001.jpg... name of your image> --custom_image_label <1 0 0 1 1... 5 original labels of your image>

### Pretrained model
* Download [G_weights.hdf5](https://drive.google.com/file/d/16n6yeQbQh4hOgobXspTU5dwEbLtnlk45/view?usp=sharing) and put into './models' directory.

## Summary
![overview](./assests/overview.PNG)

## Results (128x128)
### Women
![women](./assests/women.png)

### Men
![men](./assests/men.png)

## Related works
* [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)
* [StarGAN-Tensorflow](https://github.com/taki0112/StarGAN-Tensorflow)

## Reference
* [StarGAN paper](https://arxiv.org/abs/1711.09020)
* [Author pytorch code](https://github.com/yunjey/StarGAN)

## Author
HOANG Duc Thang
