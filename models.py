from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import keras.backend as K
import scipy
import logging
import matplotlib.pyplot as plt
import os
from keras.losses import binary_crossentropy
from DRUNet32f import get_model
import numpy as np

from utils import *
from kh_tools import *
from initial import *

smooth = 1.


def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def dice_coef_loss(y_true, y_pred):
    x = 1. - dice_coef_for_training(y_true, y_pred)
    return x


def dice_cross_loss(y_true, y_pred):
    return 0.9 * binary_crossentropy(y_true, y_pred) + 0.1 * dice_coef_loss(
        y_true, y_pred)


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    sse = 0
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
        sse += min_dist
    return idx, sse


def compute_centroids(X, idx, k):
    m, n = X.shape

    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) /
                           len(indices[0])).ravel()
    return centroids


def run_k_means(X, initial_centroids, max_iters):
    global sse
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)

    centroids = initial_centroids

    for i in range(max_iters):
        idx, sse = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return idx, centroids, sse


class Style_Model():
    def __init__(self,
                 input_height=28,
                 input_width=28,
                 output_height=28,
                 output_width=28,
                 attention_label=1,
                 is_training=True,
                 z_dim=100,
                 gf_dim=16,
                 df_dim=16,
                 c_dim=1,
                 dataset_name=None,
                 dataset_address=None,
                 input_fname_pattern=None,
                 checkpoint_dir='checkpoint',
                 log_dir='log',
                 sample_dir='sample',
                 r_alpha=0.2,
                 kb_work_on_patch=True,
                 nd_patch_size=(10, 10),
                 n_stride=1,
                 n_fetch_data=10):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection.
        :param sess: TensorFlow session.
        :param input_height: The height of image to use.
        :param input_width: The width of image to use.
        :param output_height: The height of the output images to produce.
        :param output_width: The width of the output images to produce.
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param is_training: True if in training mode.
        :param z_dim:  (optional) Dimension of dim for Z, the output of encoder. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer, i.e. g_decoder_h0. [16] 
        :param df_dim: (optional) Dimension of discrim filters in first conv layer, i.e. d_h0_conv. [16] 
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: 'UCSD', 'mnist' or custom defined name.
        :param dataset_address: path to dataset folder. e.g. './dataset/mnist'.
        :param input_fname_pattern: Glob pattern of filename of input images e.g. '*'.
        :param checkpoint_dir: path to saved checkpoint(s) directory.
        :param log_dir: log directory for training, can be later viewed in TensorBoard.
        :param sample_dir: Directory address which save some samples [.]
        :param r_alpha: Refinement parameter, trade-off hyperparameter for the G network loss to reconstruct input images. [0.2]
        :param kb_work_on_patch: Boolean value for working on PatchBased System or not, only applies to UCSD dataset [True]
        :param nd_patch_size:  Input patch size, only applies to UCSD dataset.
        :param n_stride: PatchBased data preprocessing stride, only applies to UCSD dataset.
        :param n_fetch_data: Fetch size of Data, only applies to UCSD dataset. 
        """

        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir

        self.is_training = is_training

        self.r_alpha = r_alpha
        #mnist
        # self.input_height = input_height
        # self.input_width = input_width
        # self.output_height = output_height
        # self.output_width = output_width
        #skin_32_32
        self.input_height = 32
        self.input_width = 32
        self.output_height = 32
        self.output_width = 32

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.dataset_name = dataset_name
        self.dataset_address = dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.attention_label = attention_label
        if self.is_training:
            logging.basicConfig(filename='StyleGAN_loss.log', level=logging.INFO)

        if self.dataset_name == 'mnist':
            #mnist
            # (X_train, y_train), (_, _) = mnist.load_data()
            # # Make the data range between 0~1.
            # X_train = X_train / 255
            # specific_idx = np.where(y_train == self.attention_label)[0]
            # self.data = X_train[specific_idx].reshape(-1, 28, 28, 1)
            # self.c_dim = 4

            #not mnist
            X_train = np.load("./skin_32_32_health_train.npy")
            # Make the data range between 0~1.
            X_train = X_train / 255.
            print("X_train:",X_train.shape)
            self.data = X_train
            self.c_dim = 8
        else:
            assert ('Error in loading dataset')

        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_generator(self, input_shape):
        #mnist 
        '''
        """Build the generator/R network.
        
        Arguments:
            input_shape {list} -- Generator input shape.
        
        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """
        image = Input(shape=input_shape, name='z')
        # Encoder.
        x = Conv2D(filters=self.df_dim * 2,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='g_encoder_h0_conv')(image)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 4,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='g_encoder_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 8,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='g_encoder_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder.
        # TODO: need a flexable solution to select output_padding and padding.
        # x = Conv2DTranspose(self.gf_dim*2, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=0, name='g_decoder_h0')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.gf_dim*1, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=1, name='g_decoder_h1')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.c_dim,    kernel_size = 5, strides=2, activation='tanh', padding='same', output_padding=1, name='g_decoder_h2')(x)

        x = Conv2D(self.gf_dim * 1,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim * 1,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim * 2, kernel_size=3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3,
                   kernel_size=5,
                   activation='sigmoid',
                   padding='same')(x)
        print("x:",x)
        return Model(image, x, name='R'        
        
        
        '''

        #skin_32_32
        """Build the generator/R network.
        
        Arguments:
            input_shape {list} -- Generator input shape.
        
        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """
        
        image = Input(shape=input_shape, name='z')
        # Encoder.
        x = Conv2D(filters=self.df_dim * 2,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='g_encoder_h0_conv')(image)
        
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 4,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='g_encoder_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 8,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='g_encoder_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder.
        # TODO: need a flexable solution to select output_padding and padding.
        # x = Conv2DTranspose(self.gf_dim*2, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=0, name='g_decoder_h0')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.gf_dim*1, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=1, name='g_decoder_h1')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.c_dim,    kernel_size = 5, strides=2, activation='tanh', padding='same', output_padding=1, name='g_decoder_h2')(x)
        
        x = Conv2D(self.gf_dim * 1,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)
        
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim * 1,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)
        print("x:",x)

        x = UpSampling2D((2, 2))(x)
        #mnist
        # x = Conv2D(self.gf_dim * 2, kernel_size=3, activation='relu')(x)
        x = Conv2D(self.gf_dim * 1, kernel_size=5, activation='relu', padding='same')(x)

        print("x:",x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3,
                   kernel_size=5,
                   activation='sigmoid',
                   padding='same')(x)
        return Model(image, x, name='R')

    def build_discriminator(self, input_shape):
        """Build the discriminator/D network
        
        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.
        
        Returns:
            [Tensor] -- Network output tensors.
        """

        image = Input(shape=input_shape, name='d_input')
        x = Conv2D(filters=self.df_dim,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='d_h0_conv')(image)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 2,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='d_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 4,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='d_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 8,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   name='d_h3_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

        return Model(image, x, name='D')

    def build_model(self):
        #mnist
        # image_dims = [self.input_height, self.input_width, self.c_dim]
        # d_image_dims = [self.input_height, self.input_width, self.c_dim+1]
        # labels_dims = [self.input_height, self.input_width, self.c_dim-1]
        #skin_32_32
        image_dims = [self.input_height, self.input_width, self.c_dim]
        d_image_dims = [self.input_height, self.input_width, self.c_dim+3]
        labels_dims = [self.input_height, self.input_width, self.c_dim-3]

        optimizer = RMSprop(lr=2e-4, clipvalue=1.0, decay=1e-8)
        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(d_image_dims)

        # Model to train D to discrimate real images.
        self.discriminator.compile(optimizer=optimizer,
                                   loss='binary_crossentropy')

        # Construct generator/R network.
        # self.generator = get_model(image_dims)
        self.generator = self.build_generator(image_dims)
        img = Input(shape=image_dims)
        
        reconstructed_img = self.generator(img)
        print("img:",img)
        print("before_reconstructed_img:",reconstructed_img)

        reconstructed_img_new = Concatenate()([reconstructed_img, img])
        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img_new)
        
        print("after_reconstructed_img:",reconstructed_img_new)
        # Model to train Generator/R to minimize reconstruction loss and trick D to see
        # generated images as real ones.
        self.adversarial_model = Model(img, [reconstructed_img, validity])
        self.adversarial_model.compile(
            loss=['categorical_crossentropy', 'binary_crossentropy'],
            
            # loss=[dice_cross_loss, 'binary_crossentropy'],
            loss_weights=[self.r_alpha, 1],
            optimizer=optimizer)

        print('\n\rgenerator')
        self.generator.summary()

        print('\n\rdiscriminator')
        self.discriminator.summary()

        print('\n\radversarial_model')
        self.adversarial_model.summary()

    def train(self, epochs, batch_size=128, sample_interval=500):
        # Make log folder if not exist.
        log_dir = os.path.join(self.log_dir, self.model_dir)
        os.makedirs(log_dir, exist_ok=True)

        if self.dataset_name == 'mnist':
            # Get a batch of sample images with attention_label to export as montage.
            sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        os.makedirs(self.sample_dir, exist_ok=True)
        scipy.misc.imsave(
            './{}/train_input_samples.jpg'.format(self.sample_dir),
            montage(sample_inputs[:, :, :, 0]))

        #mnist
        # cluster_data = self.data.reshape(-1, 28 * 28)
        # cluster_show = self.data.reshape(-1, 28, 28)
        #skin_32_32
        
        cluster_data = self.data.reshape(-1, 32 * 32 *3)
        cluster_show = self.data.reshape(-1, 32, 32, 3)
        
        
        if(os.path.exists("labeled_data.npy")):
            labeled_data = np.load("labeled_data.npy")
            print("exist labeled_data:",labeled_data.shape)
        else:
            if(os.path.exists("centroids.npy")):
                centroids = np.load("centroids.npy")
                print("centroids:",centroids.shape)
                idx, sse = find_closest_centroids(cluster_data, centroids)
            else:
                print("self.data:", self.data.shape[0])
                print("cluster_show:", cluster_show.shape)
                print("cluster_data:", cluster_data.shape)
                m = cluster_data.shape[0]
                cluster_points = []
                for i in range(m):
                    cluster_points.append(cluster_data[i, :])

                print("cluster_points:", len(cluster_points))
                cNum = 5
                initial_centroids = np.array(initCenters(cluster_points, m, cNum))
                print('initial_centroids:', len(initial_centroids))

                max_iters = 10
                idx, centroids, sse = run_k_means(cluster_data, initial_centroids,
                                                max_iters)
                np.save("centroids.npy", centroids)
                print("len(centroids[0]):", len(centroids[0]))
                print("centroids:", centroids)
                print("len(idx):", len(idx))
                print("sse:", sse)
                img_dir = './labels/'
                for iii in range(self.data.shape[0]):

                    scipy.misc.imsave(
                        img_dir + str(int(idx[iii])) + '/' + str(iii) + '_' +
                        str(int(idx[iii])) + '.png', cluster_show[iii])

            print("idx:", len(idx), centroids, sse)
            #mnist
            # ones = np.ones((1, 28, 28, 1))
            # zeros = np.zeros((1, 28, 28, 1))
            #skin_32_32
            ones = np.ones((1, 32, 32, 1))
            zeros = np.zeros((1, 32, 32, 1))
            label_0 =  np.concatenate((np.concatenate((ones, zeros), axis=3), zeros), axis=3)
            label_0 =  np.concatenate((label_0, zeros), axis=3)
            label_0 =  np.concatenate((label_0, zeros), axis=3)
            label_1 =  np.concatenate((np.concatenate((zeros, ones), axis=3), zeros), axis=3)
            label_1 =  np.concatenate((label_1, zeros), axis=3)
            label_1 =  np.concatenate((label_1, zeros), axis=3)
            label_2 =  np.concatenate((np.concatenate((zeros, zeros), axis=3), ones), axis=3)
            label_2 =  np.concatenate((label_2, zeros), axis=3)
            label_2 =  np.concatenate((label_2, zeros), axis=3)
            label_3 =  np.concatenate((np.concatenate((zeros, zeros), axis=3), zeros), axis=3)
            label_3 =  np.concatenate((label_3, ones), axis=3)
            label_3 =  np.concatenate((label_3, zeros), axis=3)
            label_4 =  np.concatenate((np.concatenate((zeros, zeros), axis=3), zeros), axis=3)
            label_4 =  np.concatenate((label_4, zeros), axis=3)
            label_4 =  np.concatenate((label_4, ones), axis=3)
            if (int(idx[0]) == 0):
                labels =  label_0
            if (int(idx[0]) == 1):
                labels =  label_1
            if (int(idx[0]) == 2):
                labels =  label_2
            if (int(idx[0]) == 3):
                labels =  label_3
            if (int(idx[0]) == 4):
                labels =  label_4
        
            print("self.data.shape[0]:", self.data.shape[0])
            for i in range(1, self.data.shape[0]):
                if (int(idx[i]) == 0):
                    labels = np.concatenate((labels, label_0), axis=0)
                if (int(idx[i]) == 1):
                    labels = np.concatenate((labels, label_1), axis=0)
                if (int(idx[i]) == 2):
                    labels = np.concatenate((labels, label_2), axis=0)
                if (int(idx[i]) == 3):
                    labels = np.concatenate((labels, label_3), axis=0)
                if (int(idx[i]) == 4):
                    labels = np.concatenate((labels, label_4), axis=0)
                print("labels:", labels.shape)
            print("labels:", labels.shape)
            labeled_data = np.concatenate((self.data, labels), axis=3)
            np.save("labeled_data.npy", labeled_data)
            print("self.data:", self.data.shape)
            print("labeled_data:", labeled_data.shape)

        counter = 1
        # Record generator/R network reconstruction training losses.
        plot_epochs = []
        plot_g_recon_losses = []

        # Load traning data, add random noise.
        if self.dataset_name == 'mnist':
            sample_w_noise = get_noisy_data(self.data)
            print("sample_w_noise.shape:",sample_w_noise.shape)
            print("labeled_data[3:8].shape:",labeled_data[:,:,:,3:8].shape)
            sample_w_noise = np.concatenate((sample_w_noise, labeled_data[:,:,:,3:8]), axis=3)
        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print(
                'Epoch ({}/{})-------------------------------------------------'
                .format(epoch, epochs))
            if self.dataset_name == 'mnist':
                # Number of batches computed by total number of target data / batch size.
                batch_idxs = len(self.data) // batch_size

            for idx in range(0, batch_idxs):
                # Get a batch of images and add random noise.
                
                batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                batch_noise = sample_w_noise[idx * batch_size:(idx + 1) *
                                                batch_size]
                batch_clean = self.data[idx * batch_size:(idx + 1) *
                                        batch_size]
                # Turn batch images data to float32 type.
                # batch_labels = np.array(labels).astype(np.float32)
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                batch_clean_images = np.array(batch_clean).astype(np.float32)

                # labels_batch = batch_labels[idx * batch_size:(idx + 1) * batch_size]
                labeled_batch = labeled_data[idx * batch_size:(idx + 1) * batch_size]
                labeled_batch_noise = sample_w_noise[idx * batch_size:(idx + 1) *
                                                batch_size]
                labeled_batch_clean = labeled_data[idx * batch_size:(idx + 1) *
                                        batch_size]
                # Turn batch images data to float32 type.
                labeled_batch_images = np.array(labeled_batch).astype(np.float32)
                labeled_batch_noise_images = np.array(labeled_batch_noise).astype(np.float32)
                labeled_batch_clean_images = np.array(labeled_batch_clean).astype(np.float32)
                if self.dataset_name == 'mnist':
                    batch_fake_images = self.generator.predict(
                        labeled_batch_noise_images)
                    # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
                    batch_true_images = np.concatenate((batch_clean_images, labeled_batch_clean),
                                           axis=3)
                    d_loss_real = self.discriminator.train_on_batch(
                        batch_true_images, ones)

                    batch_fake_images = np.concatenate((batch_fake_images, labeled_batch_clean),
                                           axis=3)


                    d_loss_fake = self.discriminator.train_on_batch(
                        batch_fake_images, zeros)

                    # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                    self.adversarial_model.train_on_batch(
                        labeled_batch_noise_images, [batch_clean_images, ones])
                    g_loss = self.adversarial_model.train_on_batch(
                        labeled_batch_noise_images, [batch_clean_images, ones])
                    plot_epochs.append(epoch + idx / batch_idxs)
                    plot_g_recon_losses.append(g_loss[1])
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{4:>0.3f}'.format(
                    epoch, idx, batch_idxs, d_loss_real + d_loss_fake,
                    g_loss[0], g_loss[1])
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 1000:
                    if self.dataset_name == 'mnist':
                        samples = self.generator.predict(sample_inputs)
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(
                            samples, [manifold_h, manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(
                                self.sample_dir, epoch, idx))

            # Save the checkpoint end of each epoch.
            self.save(epoch)
        # Export the Generator/R network reconstruction losses as a plot.
        plt.title('Generator/R network reconstruction losses')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs, plot_g_recon_losses)
        plt.savefig('plot_g_recon_losses.png')

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.dataset_name, self.output_height,
                                 self.output_width)

    def save(self, step):
        """Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_name = 'Style_Model_{}.h5'.format(step)
        self.adversarial_model.save_weights(
            os.path.join(self.checkpoint_dir, model_name))


if __name__ == '__main__':
    model = Style_Model(dataset_name='mnist', input_height=28, input_width=28)
    model.train(epochs=5, batch_size=128, sample_interval=500)
