import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import canny
from skimage.filters import threshold_multiotsu, gaussian, threshold_otsu
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.util import random_noise
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
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

        # Generate dataframe
        self._dataframe = None
        # data = None

        # if mode == 'FULL':
        #     self.generate_augmented_dataframe()
        #     data = self.load_data(0)
        # elif mode == 'PATCH':
        #     self.generate_patch_dataframe()
        #     data = self.load_patch_data(0)
        #
        # if self._feature_extracted == 'CANNY':
        #     data = self.load_data_canny(0)
        # elif self._feature_extracted == 'PCA':
        #     data = self.load_data_pca(0)
        # elif self._feature_extracted == 'WAV':
        #     data = self.load_data_wavelet(0)
        # elif self._feature_extracted == 'BLOB':
        #     data = self.load_data_blob(0)

        # self._dim = data.shape

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

    def generate_augmented_dataframe(self, df, name):

        # Generate the starting dataframe
        # self.generate_base_dataframe()

        # Add the pair [row, col] for each patch
        row = df.shape[0]

        dataframe = pd.DataFrame(np.repeat(df.values, 12, axis=0))
        dataframe.columns = df.columns
        df = dataframe

        transformation = [
            'V_FLIP', 'H_FLIP', 'R_NOISE',
            'ROT_30', 'ROT_60', 'ROT_90',
            # 'ROT_45', 'ROT_75', 'ROT_105',
            'ROT_120', 'ROT_150', 'ROT_210',
            # 'ROT_135', 'ROT_165', 'ROT_225',
            'ROT_240', 'ROT_270', 'ROT_300',
            # 'ROT_255', 'ROT_285', 'ROT_315'
        ]
        transformations = np.concatenate((transformation, transformation))
        for i in range(row - 2):
            transformations = np.concatenate((transformations, transformation))

        df['transformation'] = transformations

        # Save dataframe
        df.to_csv('augmented_{}_dataframe.csv'.format(name))

        return df

    def generate_patch_dataframe(self, df, name):

        # Generate the starting dataframe
        # self.generate_base_dataframe()

        # Add the pair [row, col] for each patch
        names = ['row', 'col']
        dims = [8, 10]
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

    def create_dataset(self, dataframe, loader, batch_size, shuffle=False):

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
            lambda index, label: (tf.numpy_function(loader, [dataframe, index], np.float32), label),
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

        dataframe[['label_cll', 'label_fl', 'label_mcl']] = dataframe[['label_cll', 'label_fl', 'label_mcl']].astype(
            int)
        # Extraction of data indexes of from the dataframe and labels (depending on the label names passed in input)
        data_indexes_in = list(dataframe.index)
        for i in range(len(data_indexes_in)):
            data_indexes_in[i] = str(data_indexes_in[i])

        data_indexes_out = data_indexes_in

        # Creation of the dataset with indexes and label
        dataset = tf.data.Dataset.from_tensor_slices((data_indexes_in, data_indexes_out))

        # Application of the function passed in input to every data index (from the index, data is extracted and if
        # necessary a feature is extracted with 'loader' (equal for both the data present)
        dataset = dataset.map(
            lambda index_in, index_out: (
            tf.numpy_function(loader, [index_in], np.float32), tf.numpy_function(loader, [index_out], np.float32)),
            num_parallel_calls=os.cpu_count()
        )

        # Operations for shuffling and batching of the dataset
        if shuffle:
            dataset = dataset.shuffle(len(data_indexes_in))
        # dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        return dataset

    def load_data(self, dataframe, index):

        path = dataframe.iloc[int(index)]['path']
        trans = dataframe.iloc[int(index)]['transformation']

        # Decode needed for the subsequent data loading and extraction of the correspondent file name
        if isinstance(path, bytes):
            path = path.decode()

        # Load the image the image
        image = None
        if self._color_space == "RGB":
            image = imread(path)
        elif self._color_space == "GRAY":
            image = imread(path, as_gray=True)
            image = np.expand_dims(image, axis=-1)
        elif self._color_space == "HSV":
            image = imread(path)
            image = rgb2hsv(image)

        if trans == 'V_FLIP':
            image = image[::-1, :]
        elif trans == 'H_FLIP':
            image = image[:, ::-1]
        elif trans == 'R_NOISE':
            image = random_noise(image)
        elif 'ROT' in trans.split('_')[0]:
            image = rotate(image, int(trans.split('_')[1]))
            # print(type(trans.split('_')[1]))
            # print(int(trans.split('_')[1]))

        # Resize
        scale_factor = 0.1  # percent of original size
        height = int(image.shape[0] * scale_factor)
        width = int(image.shape[1] * scale_factor)

        resized_shape = (height, width, image.shape[2])

        resized = resize(image=image, output_shape=resized_shape, preserve_range=True, anti_aliasing=True)

        # Standardize
        for i in range(resized_shape[2]):
            resized[:, :, i] = normalize(resized[:, :, i])

        return np.array(resized, dtype='float32')

    def load_patch_data(self, dataframe, index):

        path = dataframe.iloc[int(index)]['path']
        row = int(dataframe.iloc[int(index)]['row'])
        col = int(dataframe.iloc[int(index)]['col'])

        # Decode needed for the subsequent data loading and extraction of the correspondent file name
        if isinstance(path, bytes):
            path = path.decode()

        # Load the image the image
        image = None
        if self._color_space == "RGB":
            image = imread(path)
        elif self._color_space == "GRAY":
            image = imread(path, as_gray=True)
            image = np.expand_dims(image, axis=-1)
        elif self._color_space == "HSV":
            image = imread(path)
            image = rgb2hsv(image)

        # Resize
        scale_factor = 0.5  # percent of original size
        height = int(image.shape[0] * scale_factor)
        width = int(image.shape[1] * scale_factor)

        resized_shape = (height, width, image.shape[2])

        resized = resize(image=image, output_shape=resized_shape, preserve_range=True, anti_aliasing=False)

        # Standardize
        for i in range(resized_shape[2]):
            resized[:, :, i] = normalize(resized[:, :, i])

        resized = np.array(resized)

        # Extract the correct patch from the image
        # patch = image[(row - 1) * 40:(row * 40), (col - 1) * 53:(col * 53), :]
        patch = resized[(row - 1) * 64:(row * 64), (col - 1) * 64:(col * 64), :]

        return np.array(patch, dtype='float32')

    def load_data_canny(self, dataframe, index):

        # print("Extracting CANNY feature from image #{}".format(index))

        image = self.load_data(dataframe, index) if self._mode == 'FULL' else self.load_patch_data(dataframe, index)

        for i in range(image.shape[2]):
            blur = gaussian(image[:, :, i], sigma=0.4, truncate=3.5)
            low, high = threshold_multiotsu(image=blur)
            image[:, :, i] = canny(blur, sigma=1, low_threshold=low, high_threshold=high)

        return image

    def load_data_blob(self, dataframe, index):

        # print("Extracting BLOB feature from image #{}".format(index))

        image = self.load_data(dataframe, index) if self._mode == 'FULL' else self.load_patch_data(dataframe, index)

        if self._color_space == 'HSV':
            gray = image[:, :, 1]  # saturation channel
            gray = np.expand_dims(gray, -1)
        elif self._color_space == 'RGB':
            gray = rgb2gray(image)
            gray = np.expand_dims(gray, -1)
        else:
            gray = image

        thresh = threshold_otsu(image=gray)

        binarized = gray < thresh

        return np.array(binarized, dtype='float32')

    def load_data_pca(self, dataframe, index, components=32):

        # print("Extracting PCA feature from image #{}".format(index))

        image = self.load_data(dataframe, index) if self._mode == 'FULL' else self.load_patch_data(dataframe, index)

        x, y, n = image.shape

        # Extraction of the first 'components' principal components of the images
        pc_image = np.zeros([image.shape[0], components, n])
        for i in range(n):
            pca = PCA(n_components=components)  # we need K principal components.
            pc_image[:, :, i] = pca.fit_transform(image[:, :, i])

        return pc_image.astype(np.float32)

    def load_data_wavelet(self, dataframe, index):

        # print("Extracting WAVELET feature from image #{}".format(index))

        image = self.load_data(dataframe, index) if self._mode == 'FULL' else self.load_patch_data(dataframe, index)

        _, _, n = image.shape

        approx_channel = []

        if self._color_space == 'HSV':

            ca, _ = pywt.dwt(image[:, :, 0], 'sym5')  # hue
            approx_channel.append(ca)
            ca, _ = pywt.dwt(image[:, :, 2], 'sym5')  # val
            approx_channel.append(ca)

        else:

            for channel in range(n):
                # For the image coming from each channel, extract the corresponding wavelet decomposition
                ca, _ = pywt.dwt(image[:, :, channel], 'sym5')
                approx_channel.append(ca)

        n, x, y = np.array(approx_channel).shape
        approx_channel = np.array(approx_channel).reshape((x, y, n))
        # print(approx_channel.shape)

        return np.array(approx_channel)

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

    # @property
    # def dim(self):
    #     return self._dim
