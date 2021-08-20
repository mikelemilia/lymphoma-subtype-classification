import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny
from skimage.filters import threshold_multiotsu
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


class Dataset:

    def __init__(self, folder, mode, color_space, feature):

        # Parameter
        self._folder = os.path.join(os.getcwd(), folder)
        self._mode = mode
        self._color_space = color_space
        self._feature_extracted = feature

        print('Selected folder : {}'.format(self._folder))
        print('Selected mode : {}'.format(self._mode))

        self._labels = []
        self._classes = []

        self._dataframe = None

        data = None

        if mode == 'FULL':
            self.generate_dataframe()
        elif mode == 'PATCH':
            self.generate_patch_dataframe()

        if self._feature_extracted == 'CANNY':
            data = self.load_data_canny(0)
        elif self._feature_extracted == 'PCA':
            data = self.load_data_pca(0)
        elif self._feature_extracted == 'WAV':
            data = self.load_data_wavelet(0)
        elif self._feature_extracted == '':
            data = self.load_data(0) if mode == 'FULL' else self.load_patch_data(0)

        self._dim = data.shape

    def generate_dataframe(self):

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
        self._dataframe.to_csv('dataframe.csv')

    def generate_patch_dataframe(self):

        # Generate the starting dataframe
        self.generate_dataframe()

        # Add the pair [row, col] for each patch
        names = ['row', 'col']
        for k in range(2):
            total = self._dataframe.shape[0]
            dataframe = pd.DataFrame(np.repeat(self._dataframe.values, 13, axis=0))
            dataframe.columns = self._dataframe.columns
            self._dataframe = dataframe

            patch = np.arange(1, 14, 1)
            patches = np.concatenate((patch, patch))
            for i in range(total - 2):
                patches = np.concatenate((patches, patch))
            self._dataframe[names[k]] = patches

        # Save dataframe
        self._dataframe.to_csv('patch_dataframe.csv')

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

    def train_val_test_split(self):

        train, test = train_test_split(
            self._dataframe,
            test_size=0.3,
            stratify=self._dataframe[['label_cll', 'label_fl', 'label_mcl']],
            random_state=123
        )

        val, test = train_test_split(
            test,
            test_size=0.3,
            stratify=test[['label_cll', 'label_fl', 'label_mcl']],
            random_state=123
        )

        return train, val, test

    def create_dataset(self, dataframe, loader, batch_size, shuffle):

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
            lambda index, label: (tf.numpy_function(loader, [index], np.float32), label),
            num_parallel_calls=os.cpu_count()
        )

        # Operations for shuffling and batching of the dataset
        if shuffle:
            dataset = dataset.shuffle(len(data_indexes))
        dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        return dataset

    def create_dataset_nolabel(self, dataframe, loader, batch_size, shuffle):
        # Creation of the dataset of type data - data for autoencoders

        dataframe[['label_cll', 'label_fl', 'label_mcl']] = dataframe[['label_cll', 'label_fl', 'label_mcl']].astype(int)
        # Extraction of data indexes of from the dataframe and labels (depending on the label names passed in input)
        data_indexes_in = list(dataframe.index)
        for i in range(len(data_indexes_in)):
            data_indexes_in[i] = str(data_indexes_in[i])

        data_indexes_out = data_indexes_in

        # Creation of the dataset with indexes and label
        dataset = tf.data.Dataset.from_tensor_slices((data_indexes_in, data_indexes_out))

        # Application of the function passed in input to every data index (from the index, data is extracted and if
        # necessary a feature is extracted with 'function' (equal for both the data present)

        # Application of the function passed in input to every data index (from the index,
        # data is extracted and if necessary a feature is extracted with 'loader'
        dataset = dataset.map(
            lambda index_in, index_out: (tf.numpy_function(loader, [index_in], np.float32), tf.numpy_function(loader, [index_out], np.float32)),
            num_parallel_calls=os.cpu_count()
        )

        if shuffle:
            dataset = dataset.shuffle(len(data_indexes_in))
        dataset = dataset.repeat()

        # Operations for shuffling and batching of the dataset

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        return dataset

    def load_data(self, index):

        path = self._dataframe.iloc[int(index)]['path']

        # Decode needed for the subsequent data loading and extraction of the correspondent file name
        if isinstance(path, bytes):
            path = path.decode()

        image = imread(path)

        # Resize
        scale_factor = 0.3  # percent of original size
        height = int(image.shape[0] * scale_factor)
        width = int(image.shape[1] * scale_factor)

        # Pre-process the image
        if self._color_space == "GRAY":
            image = rgb2gray(image)
        elif self._color_space == "HSV":
            image = rgb2hsv(image)

        # due to the pre-process check the right number of channels
        if len(image.shape) > 2:
            resized_shape = (height, width, image.shape[2])
        else:
            resized_shape = (height, width, 1)

        resized = resize(image=image, output_shape=resized_shape, preserve_range=True, anti_aliasing=True)

        # Normalize
        resized = resized / 255.0

        return np.array(resized, dtype='float32')

    def load_data_canny(self, index):
        image = None
        if self._color_space == 'GRAY':
            image = self.load_data(index) if self._mode == 'FULL' else self.load_patch_data(index)
        elif self._color_space == 'HSV':
            image = self.load_data(index) if self._mode == 'FULL' else self.load_patch_data(index)
            image = image[:, :, 1]  # extract only Saturation channel
        elif self._color_space == 'RGB':
            image = self.load_data(index) if self._mode == 'FULL' else self.load_patch_data(index)
            image = rgb2gray(image)

        low, high = threshold_multiotsu(image=image)
        edges = canny(image, sigma=1, low_threshold=low, high_threshold=high)

        return edges

    def load_data_pca(self, index, components=32):

        image = self.load_data(index) if self._mode == 'FULL' else self.load_patch_data(index)

        x, y, n = image.shape

        # Extraction of the first 'components' principal components of the images
        pc_image = np.zeros([image.shape[0], components, n])
        for i in range(n):
            pca = PCA(n_components=components)  # we need K principal components.
            pc_image[:, :, i] = pca.fit_transform(image[:, :, i])

        # print(pc_image.astype(np.float32))
        return pc_image.astype(np.float32)

    def load_data_wavelet(self, index):

        image = self.load_data(index) if self._mode == 'FULL' else self.load_patch_data(index)

        _, _, n = image.shape

        approx_channel = []

        for channel in range(n):
            approx = []

            # For the image coming from each channel, extract the corresponding wavelet decomposition
            ca, _ = pywt.dwt(image[:, :, channel], 'sym5')
            approx.append(ca)
            approx_channel.append(approx)

        return np.array(approx_channel)

    def load_patch_data(self, index):

        path = self._dataframe.iloc[int(index)]['path']
        row = int(self._dataframe.iloc[int(index)]['row'])
        col = int(self._dataframe.iloc[int(index)]['col'])

        # Decode needed for the subsequent data loading and extraction of the correspondent file name
        if isinstance(path, bytes):
            path = path.decode()

        image = imread(path)

        # Resize
        scale_factor = 0.3  # percent of original size
        height = int(image.shape[0] * scale_factor)
        width = int(image.shape[1] * scale_factor)

        # Pre-process the image
        if self._color_space == "GRAY":
            image = rgb2gray(image)
        elif self._color_space == "HSV":
            image = rgb2hsv(image)

        # due to the pre-process check the right number of channels
        if len(image.shape) > 2:
            resized_shape = (height, width, image.shape[2])
        else:
            resized_shape = (height, width, 1)

        resized = resize(image=image, output_shape=resized_shape, preserve_range=True, anti_aliasing=False)

        # Normalize
        resized = resized / 255.0

        resized = np.array(resized)

        # Extract the correct patch from the image
        patch = resized[(row - 1) * 24:(row * 24), (col - 1) * 32:(col * 32)]

        return np.array(patch, dtype='float32')

    def random_plot(self):

        fig = plt.figure(figsize=(15, 15))
        cols = 3
        rows = 3

        # ax enables access to manipulate each of subplots
        ax = []

        for i in range(cols * rows):
            k = np.random.randint(len(self._dataframe['path']))

            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, cols, i + 1))
            ax[-1].text(0.47, 2, str(self._classes[k]), color='green')
            ax[-1].set_title("Class: " + str(self._classes[k]))  # set title

            image = imread(self._dataframe['path'][k])
            plt.imshow(image)

        plt.show()  # finally, render the plot

    @property
    def dim(self):
        return self._dim
