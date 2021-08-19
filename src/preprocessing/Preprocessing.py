from math import sqrt

from skimage.filters import threshold_multiotsu, sobel
from skimage.feature import canny, blob_log
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
import numpy as np

def extract_edges(image):

    gray = rgb2gray(image)
    low, high = threshold_multiotsu(image=gray)
    edges = canny(gray, sigma=1, low_threshold=low, high_threshold=high)

    return edges


def extract_log(image):
    gray = rgb2gray(image)

    blobs_log = blob_log(gray, max_sigma=30, num_sigma=10, threshold=.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    return blobs_log


def extract_pca(image, components=2):

    # Extraction of the first 'components' principal components of the matrices
    pc_data = np.zeros([image.shape[0], components, image.shape[2]])
    for i in range(image.shape[2]):
        pca = PCA(n_components=components)
        pc_data[:, :, i] = pca.fit_transform(image[:, :, i])
        pca = PCA(2)  # we need 2 principal components.

    print(pc_data.astype(np.float32))
    return pc_data.astype(np.float32)

def extract_wt(matrix):
    """
    Extract the approximation component obtained with wavelet decomposition of each channel of each trial inside the
    data matrix

    :param matrix: data matrix for each the wavelet will be calculated (n.trials x n.channels x n.samples)
    :return: matrix containing the approximation components of data
    """

    approx_trials = []

    for trial in matrix:
        approx = []

        for channel in trial:

            # For the signals coming from each channel, extract the corresponding wavelet decomposition

            ca, _ = pywt.dwt(channel, 'sym9')

            approx.append(ca)

        approx_trials.append(approx)

    return np.array(approx_trials)