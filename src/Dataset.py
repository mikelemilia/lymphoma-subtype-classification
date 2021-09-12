import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import canny
from skimage.filters import threshold_multiotsu
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.util import random_noise
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


class Dataset:

    def __init__(self, folder, mode, color_space, feature, is_training: bool = False, patch_size=128):

        # Parameter
        self._folder = os.path.join(os.getcwd(), folder)
        self._mode = mode
        self._color_space = color_space
        self._feature_extracted = feature
        self._is_training = is_training

        self._patch_size = patch_size

        # print('Selected folder : {}'.format(self._folder))
        # print('Selected mode : {}'.format(self._mode))
        # print('Selected color space : {}'.format(self._color_space))
        # print('Selected feature : {}'.format(self._feature_extracted))

        self._labels = []
        self._classes = []

        # Generate dataframe
        self._dataframe = None

        self._train = None
        self._val = None
        self._test = None

    def split_dataset(self):

        self._train, self._test = train_test_split(
            self._dataframe,
            test_size=0.3,
            stratify=self._dataframe[['label_cll', 'label_fl', 'label_mcl']],
            random_state=123
        )

        self._val, self._test = train_test_split(
            self._test,
            test_size=0.3,
            stratify=self._test[['label_cll', 'label_fl', 'label_mcl']],
            random_state=123
        )

        return self._train, self._val, self._test

    def get_split(self, split):
        if split == 0:
            return self._train
        elif split == 1:
            return self._val
        elif split == 2:
            return self._test

    def update_split(self, split, dataframe):
        if split == 0:
            self._train = dataframe
        elif split == 1:
            self._val = dataframe
        elif split == 2:
            self._test = dataframe

    def generate_base_dataframe(self):

        # Create dataframe
        data = []
        label_encoder = OneHotEncoder(sparse=False)

        for root, dirs, files in os.walk(self._folder):
            for file in (x for x in files if x.endswith('.tif')):
                filepath = os.path.join(root, file)

                # Populate dataframe
                data.append(filepath)
                self._classes.append(filepath.split('/')[-2])

        self._classes = np.array(self._classes).reshape(len(self._classes), 1)
        self._labels = label_encoder.fit_transform(self._classes)

        self._dataframe = pd.DataFrame(data, columns=['path'])

        self._dataframe['label_cll'] = self._labels[:, 0]
        self._dataframe['label_fl'] = self._labels[:, 1]
        self._dataframe['label_mcl'] = self._labels[:, 2]

        # Save dataframe
        self._dataframe.to_csv('base_dataframe.csv')

    def generate_augmented_dataframe(self, split, name):

        # Get dataframe
        df = self.get_split(split)

        # Add the transformation for each image
        row = df.shape[0]

        dataframe = pd.DataFrame(np.repeat(df.values, 16, axis=0))
        dataframe.columns = df.columns
        df = dataframe

        transformation = [
            '-',
            'V_FLIP', 'H_FLIP', 'R_NOISE',
            'ROT_45', 'ROT_90', 'ROT_135', 'ROT_225', 'ROT_270', 'ROT_315',
            'ROT_30', 'ROT_60', 'ROT_120', 'ROT_240', 'ROT_300', 'ROT_330',
        ]

        transformations = np.concatenate((transformation, transformation))

        for i in range(row - 2):
            transformations = np.concatenate((transformations, transformation))

        df['transformation'] = transformations

        # Save dataframe
        df.to_csv('augmented_{}_dataframe.csv'.format(name))

        # Update dataframe
        self.update_split(split, df)

        return df

    def generate_patch_dataframe(self, split, name):

        # Get dataframe
        df = self.get_split(split)

        # Column names
        names = ['row', 'col']

        # Number of vertical and horizontal split
        dims = [None, None]

        if self._patch_size == 128:
            # Patch will cover an area of 1024 * 1288 pixels w.r.t. 1040 * 1388
            dims = [8, 10]
        else:
            print("Patch dimension not yet supported!")
            exit(-1)

        for k in range(2):
            row = df.shape[0]

            dataframe = pd.DataFrame(np.repeat(df.values, dims[k], axis=0))
            dataframe.columns = df.columns
            df = dataframe

            patch = np.arange(1, dims[k] + 1, 1)
            patches = np.concatenate((patch, patch))

            for i in range(row - 2):
                patches = np.concatenate((patches, patch))

            df[names[k]] = patches

        # Save dataframe
        df.to_csv('patch_{}_dataframe.csv'.format(name))

        # Update dataframe
        self.update_split(split, df)

        return df

    def extract_tar(self, file):

        if not os.path.exists(self._folder):

            print('A dataset does not exist yet.')

            with tarfile.open(file) as tar:

                progress = tqdm(tar.getmembers())

                for member in progress:
                    tar.extract(member=member, path=self._folder)
                    progress.set_description("Extracting {}".format(member.name))

        else:

            print('Your dataset is already built!')

    def create_dataset(self, dataframe, loader, batch_size, shuffle: bool = False, split: int = 0):

        dataframe[['label_cll', 'label_fl', 'label_mcl']] = dataframe[['label_cll', 'label_fl', 'label_mcl']].astype(int)

        # Extraction of data indexes of from the dataframe and labels (depending on the label names passed in input)
        data_indexes = list(dataframe.index)
        for i in range(len(data_indexes)):
            data_indexes[i] = str(data_indexes[i])

        labels = dataframe[['label_cll', 'label_fl', 'label_mcl']]

        # Creation of the dataset with indexes and label
        dataset = tf.data.Dataset.from_tensor_slices((data_indexes, labels))

        # Application of the function passed in input to every data index (from the index,
        # data is extracted and if necessary a feature is extracted with 'loader'
        dataset = dataset.map(
            lambda index, label: (tf.numpy_function(loader, [split, index], np.float32), label),
            num_parallel_calls=os.cpu_count()
        )

        # Operations for shuffling and batching of the dataset
        if shuffle:
            dataset = dataset.shuffle(len(data_indexes))

        if self._is_training:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        return dataset

    def select_loader(self):

        # Select no-feature loader
        if self._feature_extracted == '-' and self._mode == 'FULL':
            return self.load_data
        elif self._feature_extracted == '-' and self._mode == 'PATCH':
            return self.load_patch_data

        # Select feature loader
        if self._feature_extracted == 'PCA':
            return self.load_data_pca
        elif self._feature_extracted == 'WAV':
            return self.load_data_wavelet
        elif self._feature_extracted == 'CANNY':
            return self.load_data_canny

    def preprocessing(self, image):

        # Scale pixel values : [0, 255] -> [0, 1]
        image /= 255.0

        if self._color_space == "GRAY":

            gray = rgb2gray(image)
            image = np.expand_dims(gray, axis=-1)  # Shape must be (X, Y, C)

        elif self._color_space == "HSV":

            image = rgb2hsv(image)

        # Standardize pixel value in order to get mean 0 and standard deviation 1
        mean, std = image.mean(), image.std()
        image = (image - mean) / std

        return np.array(image, dtype='float32')

    def load_data(self, split, index):

        dataframe = self.get_split(split)

        path = dataframe.iloc[int(index)]['path']
        trans = dataframe.iloc[int(index)]['transformation']

        # Decode needed for the subsequent data loading and extraction of the correspondent file name
        if isinstance(path, bytes):
            path = path.decode()

        # Load image
        image = imread(path)

        # Convert image to array
        image = np.array(image, dtype='float32')

        resized = resize(image=image, output_shape=(512, 512, image.shape[2]), preserve_range=True, anti_aliasing=True)

        # Preprocess
        filtered = self.preprocessing(resized)

        # Perform data augmentation
        if trans == 'V_FLIP':
            filtered = filtered[::-1, :]
        elif trans == 'H_FLIP':
            filtered = filtered[:, ::-1]
        elif trans == 'R_NOISE':
            filtered = random_noise(filtered, )
        elif 'ROT' in trans.split('_')[0]:
            filtered = rotate(filtered, int(trans.split('_')[1]))

        return np.array(filtered, dtype='float32')

    def load_patch_data(self, split, index):

        dataframe = self.get_split(split)

        path = dataframe.iloc[int(index)]['path']
        row = int(dataframe.iloc[int(index)]['row'])
        col = int(dataframe.iloc[int(index)]['col'])

        # Decode needed for the subsequent data loading and extraction of the correspondent file name
        if isinstance(path, bytes):
            path = path.decode()

        # Load the image the image
        image = imread(path)

        # Convert image to array
        image = np.array(image, dtype='float32')

        # Extract the correct patch 128x128 from the image
        patch = image[(row - 1) * self._patch_size:(row * self._patch_size), (col - 1) * self._patch_size:(col * self._patch_size), :]

        filtered = self.preprocessing(patch)

        return np.array(filtered, dtype='float32')

    def load_data_canny(self, split, index):

        # print("Extracting CANNY feature from image #{}".format(index))

        image = self.load_data(split, index) if self._mode == 'FULL' else self.load_patch_data(split, index)

        for i in range(image.shape[2]):
            low, high = threshold_multiotsu(image=image[:, :, i])
            image[:, :, i] = canny(image[:, :, i], sigma=2, low_threshold=low, high_threshold=high)

        return np.array(image, dtype='float32')

    def load_data_wavelet(self, split, index):

        # print("Extracting WAVELET feature from image #{}".format(index))

        image = self.load_data(split, index) if self._mode == 'FULL' else self.load_patch_data(split, index)

        if self._color_space == 'HSV':
            image = image[:, :, 2]
            image = np.expand_dims(image, axis=-1)

        approx = []

        for channel in range(image.shape[2]):
            cA, (cH, cV, cD) = pywt.dwt2(image[:, :, channel], 'sym5')
            approx.append(cA)

            # plt.figure(figsize=(30, 30))
            #
            # plt.subplot(2, 2, 1)
            # plt.imshow(cA)
            # plt.subplot(2, 2, 2)
            # plt.imshow(cH)
            # plt.subplot(2, 2, 3)
            # plt.imshow(cV)
            # plt.subplot(2, 2, 4)
            # plt.imshow(cD)

            # plt.show()

        n, x, y = np.array(approx).shape
        approx = np.array(approx).reshape((x, y, n))

        return np.array(approx, dtype='float32')

    def load_data_pca(self, split, index, components=32):

        # print("Extracting PCA feature from image #{}".format(index))
        if self._mode == 'FULL':

            image = self.load_data(split, index)
            components = 128

        else:

            image = self.load_patch_data(split, index)

        # Extraction of the first K PCs for each channel
        pc_image = np.zeros([image.shape[0], components, image.shape[2]])
        # x = np.zeros(image.shape, dtype='float32')

        for i in range(image.shape[2]):
            pca = PCA(n_components=components)
            pc_image[:, :, i] = pca.fit_transform(image[:, :, i])
            # print(f"Channel {i}: {sum(pca.explained_variance_ratio_)}")
            # x[:, :, i] = pca.inverse_transform(pc_image[:, :, i])

        return np.array(pc_image, dtype='float32')

