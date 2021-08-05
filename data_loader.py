
import os

import numpy as np
import pandas as pd

from PIL import Image

import cv2
#from sklearn.utils import shuffle
from random import shuffle
#from keras.utils import np_utils

import matplotlib.pyplot as plt

from config import Config


class All_identifier (object):
    def __init__(self, root_dir=None):
        self.root_dir = root_dir



    def load_sample(self,csv_file):



        data = pd.read_csv(os.path.join('/content/data_files',csv_file))

        data = data[['FileName','Label', 'ClassName']]

        file_names = list(data.iloc[:,0])
        labels = list(data.iloc[:, 1])
        samples = []
        for samp, lab in zip(file_names, labels):
            samples.append([samp, lab])
        return samples

    def shuffle_data(self,data):
        shuffle(data)
        return data

    def preprocessing(self,img):
        img = cv2.resize(img, (Config.resize, Config.resize))
        img = img / 255
        return img

    def data_generator(self,data,batch_size = 100, shuffle = True):

        num_samples = len(data)
        if shuffle:
            data=self.shuffle_data(data)

        while True:
            for offset in range (0,num_samples,batch_size):
                batch_samples = data[offset:offset + batch_size]
                X_train =[]
                y_train =[]
                for batch_sample in batch_samples:

                    img_name = batch_sample[0]
                    label = batch_sample[1]

                    img = cv2.imread(os.path.join(self.root_dir, 'All_images',img_name))

                    img = self.preprocessing(img)

                    X_train.append(img)
                    y_train.append(label)


                X_train= np.array(X_train)
                y_train = np.array(y_train)
                y_train = np.asarray(y_train).astype('float32').reshape((-1,1))


                yield  X_train,y_train


if __name__ == '__main__':

    dataloader = All_identifier(root_dir='/content/drive/MyDrive/CNN3')

    train_data_path = 'all_classifier_train.csv'
    test_data_path = 'all_classifier_test.csv'

    train_samples = dataloader.load_sample(train_data_path)
    test_samples = dataloader.load_sample(test_data_path)

    num_train_samples = len(train_samples)
    num_test_samples = len(test_samples)

    print('number of train samples: ', num_train_samples)
    print('number of test samples: ', num_test_samples)

    # Create generator
    batch_size = Config.batch_size
    train_datagen = dataloader.data_generator(train_samples, batch_size=batch_size)
    test_datagen = dataloader.data_generator(test_samples, batch_size=batch_size)

    for k in range(1):
        x, y = next(train_datagen)
        print('x shape: ', x.shape)
        print('label shape: ', y.shape)
        print('the label is: ', y)

    # train_samples[-15:-10]

    #### we can plot the data and see by ourselves
    fig = plt.figure(1, figsize=(12, 12))
    for i in range(8):
        plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        # x[i] = x[i][:,:,::-1] # converting BGR to RGB
        plt.imshow(x[i][:, :, ::-1], interpolation='none')
        plt.title("class_label: {}".format(y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
