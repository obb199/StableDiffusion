"""
Implementation of a Stable Diffusion Model with Unet and residual blocks with transformers.
author: Gabriel Carvalho Santana
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, change_filters, attention=False, **kwargs):
        super().__init__(**kwargs)
        self.main_convolutions = [tf.keras.layers.GroupNormalization(32),
                                  tf.keras.layers.Activation('swish'),
                                  tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same'),
                                  tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'),
                                  tf.keras.layers.GroupNormalization(32),
                                  tf.keras.layers.Activation('swish'),
                                  tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same'),
                                  tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')]

        if change_filters:
            self.sec_convolution = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')
        else:
            self.sec_convolution = lambda x: x
        
        if attention:
            self.attention_block = [tf.keras.layers.GroupNormalization(32),
                                    tf.keras.layers.MultiHeadAttention(num_heads=32, key_dim=filters, attention_axes=(1, 2))]
        else:
            self.attention_block = False
        
    def call(self, x):
        data = tf.identity(x)        
        for conv in self.main_convolutions:
            data = conv(data)
        
        if self.attention_block:
            x = self.attention_block[0](x)
            x = self.attention_block[1](x, x)
            
        return data + self.sec_convolution(x)


class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.denselayers = [tf.keras.layers.Activation('swish'),
                            tf.keras.layers.Dense(channels*4),
                            tf.keras.layers.Activation('swish'),
                            tf.keras.layers.Dense(channels),
                            tf.keras.layers.Reshape([1, 1, channels])]
        
    def call(self, x):
        for layer in self.denselayers:
            x = layer(x)
        
        return x

class ProcessBlock(tf.keras.layers.Layer):
    def __init__(self, filters, change_filters, attention=False, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = TimeEmbedding(filters)
        self.residual1       = ResidualBlock(filters, True, attention)
        self.residual2       = ResidualBlock(filters, change_filters, attention)
        self.residual3       = ResidualBlock(filters, change_filters, attention)
        self.residual4       = ResidualBlock(filters, change_filters, attention)

    def call(self, x):
        x[2] = self.embedding_layer(x[2])
        x[0] = self.residual1(x[0])
        x[1] = x[0] + self.residual2(x[1]) + self.residual3(x[1])*x[2]
        return self.residual4(x[1])


class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
    
        self.time_process = [tf.keras.layers.Dense(1024),
                             tf.keras.layers.Activation('swish'),
                             tf.keras.layers.Dense(1024)]

        self.init_conv = [tf.keras.layers.Conv2D(32, kernel_size=1, padding='same'),
                          ResidualBlock(64, True)]
        
        self.init_seed = [tf.keras.layers.Conv2D(32, kernel_size=1, padding='same'),
                          ResidualBlock(64, True)]

        # ---------------- left  ---------------
        self.left = [ProcessBlock(128, True),
                     ProcessBlock(128, False),
                     ProcessBlock(128, False),
                     ProcessBlock(128, False),
                     ProcessBlock(128, False),
                     ProcessBlock(128, False),
                     ProcessBlock(256, True, True),
                     ProcessBlock(256, False, True),
                     ProcessBlock(256, True, True)]
        
        self.stride_convs = [tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides=2),
                             tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides=2),
                             tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides=2),
                             tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides=2)]
        
        # --------------- mid ----------------
        self.mid = [ResidualBlock(512, True),
                    ResidualBlock(512, False)]

        # ----------------right----------------
        self.right = [ProcessBlock(256, True, True),
                      ProcessBlock(256, True, True),
                      ProcessBlock(256, True, True),
                      ProcessBlock(128, True),
                      ProcessBlock(128, True),
                      ProcessBlock(128, True),
                      ProcessBlock(128, True),
                      ProcessBlock(128, True),
                      ProcessBlock(128, True)]
        
        self.stride_transp = [tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, padding='same', strides=2),
                              tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, padding='same', strides=2),
                              tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, padding='same', strides=2),
                              tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, padding='same', strides=2)]
        

        self.concat = tf.keras.layers.Concatenate()

        self.end = [tf.keras.layers.GroupNormalization(32),
                    tf.keras.layers.Activation('swish'),
                    tf.keras.layers.Conv2D(3, kernel_size=1, padding='same')]
    
    def call(self, x):
        seed, x_input, x_ts = x[0], x[1], x[2]
        left_outputs = []
        
        for layer in self.time_process:
            x_ts = layer(x_ts)
        x_input = self.init_conv[1](self.init_conv[0](x_input))
        seed    = self.init_seed[1](self.init_seed[0](seed))
        
        z = 0
        for idx in range(0, len(self.left), 3):
            if idx != 0:
                x_input = self.stride_convs[z](x_input)
                seed    = self.stride_convs[z+1](seed)
                z += 2
                
            x_input = self.left[idx]([seed, x_input, x_ts])
            left_outputs.append(x_input)
            x_input = self.left[idx+1]([seed, x_input, x_ts])
            left_outputs.append(x_input)
            x_input = self.left[idx+2]([seed, x_input, x_ts])
            left_outputs.append(x_input)
        
        x = self.mid[0](x_input)
        for layer in self.mid[1:]:
            x = layer(x)
        z = 0
        for idx in range(0, len(self.right), 3):
            if idx != 0:
                x = self.stride_transp[z](x)
                seed = self.stride_transp[z+1](seed)
                z += 2
                
            k = self.concat([x, left_outputs.pop()])
            x = self.right[idx]([seed, k, x_ts])
            k = self.concat([x, left_outputs.pop()])
            x = self.right[idx+1]([seed, k, x_ts])
            k = self.concat([x, left_outputs.pop()])
            x = self.right[idx+2]([seed, k, x_ts])
        
        for end_layers in self.end:
            x = end_layers(x)
        return x
    
    def complete_images(self, seed, h, w):
        noise = np.random.random(size=(h*w, IMG_DIMS[1], IMG_DIMS[0]//2, 3))
        for i in trange(timesteps):
            noise = self.predict([seed, noise, np.reshape(np.full((h*w,), i), (-1,1))], verbose=0, batch_size=16)
        return np.concatenate([seed, noise], axis=2)

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size, shuffle=True):
        self.data        = data # array of strings with original images name with directory
        self.batch_size      = batch_size #images per batch
        self.shuffle         = shuffle # true or false to shuffle data after any epochs
        self.on_epoch_end() # call of the function

    
    def __forward_noise(self, x, t):
        a = time_bar[t]      # base on t
        b = time_bar[t + 1]  # image for t + 1

        noise = np.random.random(size=(x.shape[0], IMG_DIMS[1], IMG_DIMS[0]//2, 3))  # noise mask
        a = a.reshape((-1, 1, 1, 1))
        b = b.reshape((-1, 1, 1, 1))
        img_a = x * (1 - a) + noise * a
        img_b = x * (1 - b) + noise * b
        return img_a, img_b
    
    def __generate_ts(self, num):
        return np.random.randint(0, timesteps, size=num)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        indexes_for_batch = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes_for_batch)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_for_batch):
        'Generates data containing batch_size samples'
        images_for_train = np.array([self.data[i] for i in indexes_for_batch])
        timesteps = self.__generate_ts(self.batch_size)
        imgs_t1, imgs_t2 = self.__forward_noise(images_for_train[:,:,IMG_DIMS[0]//2:,:], timesteps)
        X = [images_for_train[:,:,0:IMG_DIMS[0]//2,:], imgs_t1, np.array(timesteps).reshape(-1, 1)]
        y = imgs_t2
        return X, y

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, dim_img, shuffle=True):
        self.list_IDs        = list_IDs # array of strings with original images name with directory
        self.batch_size      = batch_size #images per batch
        self.dim_img         = dim_img # tuple with width and height of image like (192, 256)
        self.shuffle         = shuffle # true or false to shuffle data after any epochs
        self.on_epoch_end() # call of the function

    
    def __forward_noise(self, x, t):
        a = time_bar[t]      # base on t
        b = time_bar[t + 1]  # image for t + 1

        noise = np.random.random(size=(x.shape[0], IMG_SIZE, IMG_SIZE//2, 3))  # noise mask
        a = a.reshape((-1, 1, 1, 1))
        b = b.reshape((-1, 1, 1, 1))
        img_a = x * (1 - a) + noise * a
        img_b = x * (1 - b) + noise * b
        return img_a, img_b
    
    def __generate_ts(self, num):
        return np.random.randint(0, timesteps, size=num)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        indexes_for_batch = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes_for_batch)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_for_batch):
        'Generates data containing batch_size samples'
        images_for_train = [cv2.resize(plt.imread(directory + self.list_IDs[i]), (self.dim_img, self.dim_img)) for i in indexes_for_batch]
        timesteps = self.__generate_ts(self.batch_size)
        imgs_t1, imgs_t2 = self.__forward_noise(np.array(images_for_train)[:,:,self.dim_img//2:,:], timesteps)
        X = [np.array(images_for_train)[:,:,0:self.dim_img//2,:], imgs_t1, np.array(timesteps).reshape(-1, 1)]
        y = imgs_t2

        return X, y
