import numpy as np

from models import Style_Model
import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.datasets import mnist
import seaborn as sns
import pandas as pd
from matplotlib.colors import hsv_to_rgb


def find_closest_centroids_cluster(X, centroids):
    m = X.shape[0]
    print("X.shape:",X.shape)
    
    X = X.reshape(-1,32*32*3)
    k = centroids.shape[0]
    idx = np.zeros(m)
    sse = 0
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            print("dist:",j,dist)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
                # cluster = j
        sse += min_dist
    return idx, sse

attention_label = 7

self =Style_Model(dataset_name='mnist', input_height=32,input_width=32)
self.adversarial_model.load_weights('./checkpoint/Style_Model_5.h5')
# (X_train, y_train), (_, _) = mnist.load_data()
# X_train = X_train / 255.


# print("X_train:",X_train.shape)
X_train = np.load("./skin_32_32_lesion_test_hsv.npy")
X_train = X_train / 255.
# Make the data range between 0~1.

test_data = X_train[0:200]
print("test_data:",test_data.shape)
# specific_idx = np.where(y_train == attention_label)
# test_data = X_train[specific_idx].reshape(-1, 28, 28, 1)
# test_data = test_data[0:200]
centroids = np.load("skin_32_32_centroids_hsv.npy")
print("centroids:",centroids.shape)
idx, sse = find_closest_centroids_cluster(test_data, centroids)


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
print("test_data[0]:", test_data.shape[0])
for i in range(1, test_data.shape[0]):
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
print("test_data:",test_data.shape)
test_data = np.concatenate((test_data, labels), axis=3)
print("test_data:", test_data.shape)
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
        fig= plt.figure(figsize=(8, 8))
        fig.add_subplot(rows, columns, 1)
        input_image = data.reshape((32, 32, 8))


        reconstructed_image = model_predicts[0].reshape((32, 32, 3))
        input_image = data[:,:,:,0:3].reshape(32, 32,3)
        # input_image = X_train[0:10]
        reconstructed_image = model_predicts[0].reshape(32, 32, 3)
        plt.title('Input')
        input_image_plt = hsv_to_rgb(input_image)
        plt.imshow(input_image_plt, label='Input')
        fig.add_subplot(rows, columns, 2)
        plt.title('Reconstruction')
        reconstructed_image_plt = hsv_to_rgb(reconstructed_image)
        plt.imshow(reconstructed_image_plt, label='Reconstructed')
        plt.show()
        y_true = K.variable(reconstructed_image)
        y_pred = K.variable(input_image)
        error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
        print('Reconstruction loss, Discriminator Output:', error, model_predicts[1][0][0])
        # print(error+1-model_predicts[1][0][0])



test_reconstruction()
