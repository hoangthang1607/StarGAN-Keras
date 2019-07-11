from __future__ import print_function, division
import numpy as np
import os
import cv2
from PIL import Image
import random
from functools import partial
from utils import *

import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers.merge import _Merge
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, ReLU, UpSampling2D
from keras.layers import Reshape, Dropout, Concatenate, Lambda, Multiply, Add, Flatten, Dense
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras import backend as K

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def define_batch_size(self, bs):
        self.bs = bs
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.bs, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class StarGAN(object):
    def __init__(self, args):

        # Model configuration.
        self.c_dim = args.c_dim
        self.image_size = args.image_size
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_repeat_num = args.d_repeat_num
        self.lambda_cls = args.lambda_cls
        self.lambda_rec = args.lambda_rec
        self.lambda_gp = args.lambda_gp

        # Training configuration.
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_iters = args.num_iters
        self.num_iters_decay = args.num_iters_decay
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.n_critic = args.n_critic
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.selected_attrs = args.selected_attrs

        # Test configurations.
        self.test_iters = args.test_iters

        # Miscellaneous.
        self.mode = args.mode 

        # Directories.
        self.data_dir = args.data_dir
        self.sample_dir = args.sample_dir
        self.model_save_dir = args.model_save_dir
        self.result_dir = args.result_dir

        # Step size.
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.lr_update_step = args.lr_update_step

        # Custom image
        self.custom_image_name = args.custom_image_name
        self.custom_image_label = args.custom_image_label

    def ResidualBlock(self, inp, dim_out):
        """Residual Block with instance normalization."""
        x = ZeroPadding2D(padding = 1)(inp)
        x = Conv2D(filters = dim_out, kernel_size=3, strides=1, padding='valid', use_bias = False)(x)
        x = InstanceNormalization(axis = -1)(x)
        x = ReLU()(x)
        x = ZeroPadding2D(padding = 1)(x)
        x = Conv2D(filters = dim_out, kernel_size=3, strides=1, padding='valid', use_bias = False)(x)
        x = InstanceNormalization(axis = -1)(x)
        return Add()([inp, x])

    def build_generator(self):
        """Generator network."""
        # Input tensors
        inp_c = Input(shape = (self.c_dim, ))
        inp_img = Input(shape = (self.image_size, self.image_size, 3))
    
        # Replicate spatially and concatenate domain information
        c = Lambda(lambda x: K.repeat(x, self.image_size**2))(inp_c)
        c = Reshape((self.image_size, self.image_size, self.c_dim))(c)
        x = Concatenate()([inp_img, c])
    
        # First Conv2D
        x = Conv2D(filters = self.g_conv_dim, kernel_size = 7, strides = 1, padding = 'same', use_bias = False)(x)
        x = InstanceNormalization(axis = -1)(x)
        x = ReLU()(x)
    
        # Down-sampling layers
        curr_dim = self.g_conv_dim
        for i in range(2):
            x = ZeroPadding2D(padding = 1)(x)
            x = Conv2D(filters = curr_dim*2, kernel_size = 4, strides = 2, padding = 'valid', use_bias = False)(x)
            x = InstanceNormalization(axis = -1)(x)
            x = ReLU()(x)
            curr_dim = curr_dim * 2
        
        # Bottleneck layers.
        for i in range(self.g_repeat_num):
            x = self.ResidualBlock(x, curr_dim)
        
        # Up-sampling layers
        for i in range(2):
            x = UpSampling2D(size = 2)(x)       
            x = Conv2D(filters = curr_dim // 2, kernel_size = 4, strides = 1, padding = 'same', use_bias = False)(x)
            x = InstanceNormalization(axis = -1)(x)
            x = ReLU()(x)        
            curr_dim = curr_dim // 2
    
        # Last Conv2D
        x = ZeroPadding2D(padding = 3)(x)
        out = Conv2D(filters = 3, kernel_size = 7, strides = 1, padding = 'valid', activation = 'tanh', use_bias = False)(x)
    
        return Model(inputs = [inp_img, inp_c], outputs = out)        

    def build_discriminator(self):
        """Discriminator network with PatchGAN."""
        inp_img = Input(shape = (self.image_size, self.image_size, 3))
        x = ZeroPadding2D(padding = 1)(inp_img)
        x = Conv2D(filters = self.d_conv_dim, kernel_size = 4, strides = 2, padding = 'valid', use_bias = False)(x)
        x = LeakyReLU(0.01)(x)
    
        curr_dim = self.d_conv_dim
        for i in range(1, self.d_repeat_num):
            x = ZeroPadding2D(padding = 1)(x)
            x = Conv2D(filters = curr_dim*2, kernel_size = 4, strides = 2, padding = 'valid')(x)
            x = LeakyReLU(0.01)(x)
            curr_dim = curr_dim * 2
    
        kernel_size = int(self.image_size / np.power(2, self.d_repeat_num))
    
        out_src = ZeroPadding2D(padding = 1)(x)
        out_src = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'valid', use_bias = False)(out_src)
    
        out_cls = Conv2D(filters = self.c_dim, kernel_size = kernel_size, strides = 1, padding = 'valid', use_bias = False)(x)
        out_cls = Reshape((self.c_dim, ))(out_cls)
    
        return Model(inp_img, [out_src, out_cls])

    def classification_loss(self, Y_true, Y_pred) :
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_true, logits=Y_pred))

    def wasserstein_loss(self, Y_true, Y_pred):
        return K.mean(Y_true*Y_pred)

    def reconstruction_loss(self, Y_true, Y_pred):
        return K.mean(K.abs(Y_true - Y_pred))

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def build_model(self):
        self.G = self.build_generator()
        self.D = self.build_discriminator()

        # First don't update weights of Generator block
        self.G.trainable = False

        # Compute output with real images.
        x_real = Input(shape = (self.image_size, self.image_size, 3))
        out_src_real, out_cls_real = self.D(x_real)

        # Compute output with fake images.
        label_trg = Input(shape = (self.c_dim, ))
        x_fake = self.G([x_real, label_trg])
        out_src_fake, out_cls_fake = self.D(x_fake)

        # Compute output for gradient penalty.
        rd_avg = RandomWeightedAverage()
        rd_avg.define_batch_size(self.batch_size)
        x_hat = rd_avg([x_real, x_fake])
        out_src, _ = self.D(x_hat)
            
        # Use Python partial to provide loss function with additional 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=x_hat)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        # Define training model D
        self.train_D = Model([x_real, label_trg], [out_src_real, out_cls_real, out_src_fake, out_src])

        # Setup loss for train_D
        self.train_D.compile(loss = [self.wasserstein_loss, self.classification_loss, self.wasserstein_loss, partial_gp_loss], 
                             optimizer=Adam(lr = self.d_lr, beta_1 = self.beta1, beta_2 = self.beta2), loss_weights = [1, self.lambda_cls, 1, self.lambda_gp])

        # Update G and not update D
        self.G.trainable = True
        self.D.trainable = False

        # All inputs
        real_x = Input(shape = (self.image_size, self.image_size, 3))
        org_label = Input(shape = (self.c_dim, ))
        trg_label = Input(shape = (self.c_dim, ))

        # Compute output of fake image
        fake_x = self.G([real_x, trg_label])
        fake_out_src, fake_out_cls = self.D(fake_x)

        # Target-to-original domain.
        x_reconst = self.G([fake_x, org_label])

        # Define traning model G
        self.train_G = Model([real_x, org_label, trg_label], [fake_out_src, fake_out_cls, x_reconst])

        # Setup loss for train_G
        self.train_G.compile(loss = [self.wasserstein_loss, self.classification_loss, self.reconstruction_loss], 
                             optimizer=Adam(lr = self.g_lr, beta_1 = self.beta1, beta_2 = self.beta2), loss_weights = [1, self.lambda_cls, self.lambda_rec])
        
        """ Input Image"""
        self.Image_data_class = ImageData(data_dir=self.data_dir, selected_attrs=self.selected_attrs)
        self.Image_data_class.preprocess()

    def train(self):
        data_iter = get_loader(self.Image_data_class.train_dataset, self.Image_data_class.train_dataset_label, self.Image_data_class.train_dataset_fix_label, 
                               image_size=self.image_size, batch_size=self.batch_size, mode=self.mode)

        # Training
        valid = -np.ones((self.batch_size, 2, 2, 1))
        fake =  np.ones((self.batch_size, 2, 2, 1))
        dummy = np.zeros((self.batch_size, 2, 2, 1)) # Dummy gt for gradient penalty
        for epoch in range(self.num_iters):
            imgs, orig_labels, target_labels, fix_labels, _ = next(data_iter)
    
            # Setting learning rate (linear decay)
            if epoch > (self.num_iters - self.num_iters_decay):
                K.set_value(self.train_D.optimizer.lr, self.d_lr*(self.num_iters - epoch)/(self.num_iters - self.num_iters_decay))
                K.set_value(self.train_G.optimizer.lr, self.g_lr*(self.num_iters - epoch)/(self.num_iters - self.num_iters_decay))
    
            # Training Discriminators        
            D_loss = self.train_D.train_on_batch(x = [imgs, target_labels], y = [valid, orig_labels, fake, dummy])
        
            # Training Generators
            if (epoch + 1) % self.n_critic == 0:
                G_loss = self.train_G.train_on_batch(x = [imgs, orig_labels, target_labels], y = [valid, target_labels, imgs])
        
            if (epoch + 1) % self.log_step == 0:
                print(f"Iteration: [{epoch + 1}/{self.num_iters}]")
                print(f"\tD/loss_real = [{D_loss[1]:.4f}], D/loss_fake = [{D_loss[3]:.4f}], D/loss_cls =  [{D_loss[2]:.4f}], D/loss_gp = [{D_loss[4]:.4f}]")
                print(f"\tG/loss_fake = [{G_loss[1]:.4f}], G/loss_rec = [{G_loss[3]:.4f}], G/loss_cls = [{G_loss[2]:.4f}]") 

            if (epoch + 1) % self.model_save_step == 0:  
                self.G.save_weights(os.path.join(self.model_save_dir, 'G_weights.hdf5'))
                self.D.save_weights(os.path.join(self.model_save_dir, 'D_weights.hdf5'))
                self.train_D.save_weights(os.path.join(self.model_save_dir, 'train_D_weights.hdf5'))
                self.train_G.save_weights(os.path.join(self.model_save_dir, 'train_G_weights.hdf5'))    

    def test(self):
        G_weights_dir = os.path.join(self.model_save_dir, 'G_weights.hdf5')
        if not os.path.isfile(G_weights_dir):
            print("Don't find weight's generator model")
        else:
            self.G.load_weights(G_weights_dir)

        data_iter = get_loader(self.Image_data_class.test_dataset, self.Image_data_class.train_dataset_label, self.Image_data_class.train_dataset_fix_label, 
                               image_size=self.image_size, batch_size=self.batch_size, mode=self.mode)        
        n_batches = int(len(self.sample_step) / self.batch_size)
        total_samples = n_batches * self.batch_size

        for i in range(n_batches):
            imgs, orig_labels, target_labels, fix_labels, names = next(data_iter)
            for j in range(self.batch_size):
                preds = self.G.predict([np.repeat(np.expand_dims(imgs[j], axis = 0), len(self.selected_attrs), axis = 0), fix_labels[j]])
                for k in range(len(self.selected_attrs)):                    
                    Image.fromarray((preds[k]*127.5 + 127.5).astype(np.uint8)).save(os.path.join(self.result_dir, names[j].split(os.path.sep)[-1].split('.')[0] + f'_{k + 1}.png'))

    def custom(self):
        G_weights_dir = os.path.join(self.model_save_dir, 'G_weights.hdf5')
        if not os.path.isfile(G_weights_dir):
            print("Don't find weight's generator model")
        else:
            self.G.load_weights(G_weights_dir)                        

        path = os.path.join(self.sample_dir, self.custom_image_name)
        target_list = create_labels([self.custom_image_label], selected_attrs=self.selected_attrs)[0]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_keep_aspect_ratio(image, width = self.image_size, height = self.image_size)
        image = np.array([image])/127.5 - 1
        preds = self.G.predict([np.repeat(image, len(self.selected_attrs), axis = 0), target_list])
        for k in range(len(self.selected_attrs)):                    
            Image.fromarray((preds[k]*127.5 + 127.5).astype(np.uint8)).save(os.path.join(self.sample_dir, self.custom_image_name.split('.')[0] + f'_{k + 1}.png'))        