import numpy as np

from models import Style_Model
import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.datasets import mnist
import seaborn as sns
import pandas as pd

attention_label = 7

self =Style_Model(dataset_name='mnist', input_height=28,input_width=28)
self.adversarial_model.load_weights('./checkpoint/Style_Model_4.h5')
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train / 255.

# print("X_train:",X_train.shape)
specific_idx = np.where(y_train == attention_label)
test_data = X_train[specific_idx].reshape(-1, 28, 28, 1)
# print("X_train[specific_idx]:",X_train[specific_idx].shape)
# cluster_data = X_train[specific_idx].reshape(-1, 28*28)
# cluster_show = X_train[specific_idx].reshape(-1, 28, 28)


# print("test_data:",test_data.shape[0])
# c_dim = 2


# print("cluster_data:",cluster_data.shape)
# cluster_data2 = pd.DataFrame(cluster_data)
# m = cluster_data.shape[0]
# cluster_points = []
# for i in range(m):
#     cluster_points.append(cluster_data[i,:])

# print("cluster_points:",len(cluster_points))

# incenter= np.array(initCenters(cluster_points, m, 3))
# print('incenter:',len(incenter))

# initial_centroids = incenter
# idx, centroids, sse = run_k_means(cluster_data, initial_centroids, 3)
# print("idx:", len(idx), centroids, sse)
# ones  = np.ones((1, 28, 28, 1))
# labels = int(idx[0])*ones 
# print("labels:", idx[i])
# for i in range(1, test_data.shape[0]):
#     labels = np.concatenate((labels, idx[i]*ones),axis = 0)
#     print("labels:", idx[i], labels.shape)
# print("labels:",labels.shape)
# test_data = np.concatenate((test_data,labels),axis=3)
# print("test_data:",test_data.shape)

def test_reconstruction():
   
    for i in range(test_data.shape[0]):
        data = test_data[i:i+1]
        model_predicts = self.adversarial_model.predict(data)
        # print("data:",data.shape)
        # print("model_predicts:",model_predicts[0].shape)
        columns = 1
        rows = 2
        # fig= plt.figure(figsize=(8, 8))
        # fig.add_subplot(rows, columns, 1)
        input_image = data[:,:,:,0].reshape((28, 28))
        reconstructed_image = model_predicts[0][:,:,:,0].reshape((28, 28))
        # plt.title('Input')
        # plt.imshow(input_image, label='Input')
        # fig.add_subplot(rows, columns, 2)
        # plt.title('Reconstruction')
        # plt.imshow(reconstructed_image, label='Reconstructed')
        # plt.show()
        y_true = K.variable(reconstructed_image)
        y_pred = K.variable(input_image)
        error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
        # print('Reconstruction loss, Discriminator Output:', error,model_predicts[1][0][0])
        print(error+1-model_predicts[1][0][0])



test_reconstruction()