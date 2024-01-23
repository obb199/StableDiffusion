"""
Implementation of a Stable Diffusion Model with Unet and residual blocks.
author: Gabriel Carvalho Santana
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.main_convolutions = [tf.keras.layers.GroupNormalization(),
                                  tf.keras.layers.Activation('swish'),
                                  tf.keras.layers.Conv2D(channels, kernel_size=1, padding='same'),
                                  tf.keras.layers.Conv2D(channels, kernel_size=3, padding='same')]

        self.sec_convolution = tf.keras.layers.Conv2D(channels, kernel_size=1, padding='same')

    def call(self, x):
        data = tf.identity(x)

        for conv in self.main_convolutions:
            data = conv(data)

        if data.shape[-1] != x.shape[-1]:
            x = self.sec_convolution(x)

        return data + x


class ImageTimeProcessBlock(tf.keras.layers.Layer):
    def __init__(self, process_dimension, **kwargs):
        super().__init__(**kwargs)
        self.image_process = ResidualBlock(process_dimension)

        self.time_process = [tf.keras.layers.Dense(process_dimension),
                             tf.keras.layers.LayerNormalization(),
                             tf.keras.layers.Activation("swish"),
                             tf.keras.layers.Dense(process_dimension),
                             tf.keras.layers.LayerNormalization(),
                             tf.keras.layers.Activation("swish"),
                             tf.keras.layers.Reshape([1, 1, process_dimension])]

        self.image_time_process = [ResidualBlock(process_dimension),
                                   tf.keras.layers.LayerNormalization(),
                                   tf.keras.layers.Activation('swish')]

    def call(self, x):
        image, time = x[0], x[1]

        image = self.image_process(image)

        for layer in self.time_process:
            time = layer(time)

        output = self.image_time_process[0](x[0])
        output += image * time
        output = self.image_time_process[1](output)
        output = self.image_time_process[2](output)

        return output


class UNet(tf.keras.layers.Layer):
    def __init__(self, input_shape, init_conv_filters, dim_embedding, unet_filters, **kwargs):
        super().__init__(**kwargs)

        self.first_conv = tf.keras.layers.Conv2D(init_conv_filters, kernel_size=3, padding='same')
        self.last_conv = tf.keras.layers.Conv2D(input_shape[-1], kernel_size=1, padding='same')

        self.init_input_process = [tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(dim_embedding),
                                   tf.keras.layers.LayerNormalization(),
                                   tf.keras.layers.Activation('swish')]

        self.left_layers = [ImageTimeProcessBlock(c) for c in unet_filters]
        self.right_layers = [ImageTimeProcessBlock(c) for c in unet_filters]

        self.mid_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                input_shape[0] // (2 ** (len(unet_filters))) * input_shape[1] // (2 ** (len(unet_filters))) *
                unet_filters[-1]),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Reshape(
                [input_shape[0] // (2 ** (len(unet_filters))), input_shape[1] // (2 ** (len(unet_filters))),
                 unet_filters[-1]])
        ]

        self.avgpool = tf.keras.layers.MaxPooling2D(padding='same')
        self.upsampling = tf.keras.layers.UpSampling2D()

    def call(self, x):
        image, time = x[0], x[1]
        image = self.first_conv(image)

        for init_process_layer in self.init_input_process:
            time = init_process_layer(time)

        left_res = []

        for left_layer in self.left_layers:
            image = left_layer([image, time])
            image = self.avgpool(image)
            left_res.append(image)

        for mid_layer in self.mid_layers:
            image = mid_layer(image)

        i = -1
        for right_layer in self.right_layers:
            image = tf.concat([image, left_res[i]], -1)
            image = right_layer([image, time])
            image = self.upsampling(image)
            i -= 1

        return self.last_conv(image)


class StableDiffusion(tf.keras.Model):
    def __init__(self,
                 image_shape,
                 init_conv_filters,
                 dim_embedding,
                 unet_filters,
                 timestep=15,
                 batch_size=16,
                 lr_decay=0.98,
                 shuffle_data=True,
                 save_weights=False,
                 file_path_weights='',
                 **kwargs):

        super().__init__(**kwargs)
        self.model = UNet(image_shape,
                          init_conv_filters,
                          dim_embedding,
                          unet_filters,
                          **kwargs)

        self.image_shape = image_shape
        self.timestep = timestep
        self.timebar = 1 - np.linspace(0, 1.0, timestep + 1)
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.shuffle_data = shuffle_data
        self.save_weights = save_weights
        self.fiel_path_weights = file_path_weights

    def call(self, x):
        return self.model(x)

    def forward_noise(self, x, t: int):
        t1 = self.timebar[t].reshape((-1, 1, 1, 1))  # base on t
        t2 = self.timebar[t + 1].reshape((-1, 1, 1, 1))  # image for t + 1

        noise = np.random.normal(
            size=[self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]])  # noise mask

        img_a = x * (1 - t1) + noise * (t1)
        img_b = x * (1 - t2) + noise * (t2)

        return img_a, img_b

    def train(self, R=10):
        def train_one(x_img):
            x_ts = np.random.randint(0, TIMESTEPS, size=len(x_img))
            x_a, x_b = self.forward_noise(x_img, x_ts)
            loss = self.train_on_batch([x_a, x_ts], x_b)
            return loss

        bar = trange(R)
        for i in bar:
            for j in range(0, len(dataset), self.batch_size):
                if j + BATCH_SIZE < len(dataset):
                    image_batch = dataset[j:j + self.batch_size]
                    loss_image_batch = train_one(image_batch)
                    pg = (j / BATCH_SIZE)
                    bar.set_description(f'loss: {loss_image_batch:.5f}, batch: {pg}')

            model.optimizer.learning_rate = model.optimizer.learning_rate * self.lr_decay
            np.random.shuffle(dataset)

    def generate_samples(self, n_creations: int):
        gens = np.random.normal(size=(n_creations, IMG_SIZE[0], IMG_SIZE[1], 3))
        for i in trange(TIMESTEPS):
            gens = self.predict([gens, np.full((n_creations), i)], verbose=0)
        show_examples(gens, n_creations // 5, 5)

    def generate_sample_steps(self):
        xs = []
        x = np.random.normal(size=(TIMESTEPS, IMG_SIZE[0], IMG_SIZE[1], 3))

        for i in trange(TIMESTEPS):
            t = i
            x = self.predict([x, np.full((TIMESTEPS), t)], verbose=0)
            xs.append(x[0])

        plt.figure(figsize=(20, 20))
        for i in range(len(xs)):
            plt.subplot(10, 10, i + 1)
            plt.imshow((xs[i] - xs[i].min()) / (xs[i].max() - xs[i].min()))
            plt.title(f'{i}')
            plt.axis('off')





