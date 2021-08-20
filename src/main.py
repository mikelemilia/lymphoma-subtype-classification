import argparse
import sys

from Dataset import *
from models import CNN, DNN, DCNN, AEDNN


def parse_input():

    f = ''
    a = ''
    m = ''
    c = ''
    e = ''

    # Create parser
    parser = argparse.ArgumentParser(description='3-CLASS LYMPHOMA SUBTYPE CLASSIFICATION')

    # Add arguments
    parser.add_argument('-f', '--folder', type=str, default='data', help='dataset folder path')
    parser.add_argument('-a', '--arch', type=str, default='CNN', help='architecture, it can be CNN, DNN, D-CNN, AE-DNN')
    parser.add_argument('-m', '--mode', type=str, default='FULL', help='preprocessing mode, it can be FULL or PATCH')
    parser.add_argument('-c', '--color', type=str, default='RGB', help='color space used, it can be RGB, GRAY, HSV')
    parser.add_argument('-e', '--extra', type=str, default='-', help='extracted features, it can be CANNY, PCA, BLOB')


    # Retrieve arguments value
    args = parser.parse_args()

    # Check retrieved arguments
    if not os.path.exists(args.folder):
        print('You must provide an existing data location', file=sys.stderr)
        exit(-1)
    else:
        f = args.folder

    if str(args.arch).upper() not in ['CNN', 'DNN', 'D-CNN', 'AE-DNN', 'RNN']:
        print('You must select a valid implemented architecture. Valid options are: CNN, D-CNN, AE-DNN.', file=sys.stderr)
        exit(-1)
    else:
        a = str(args.arch).upper()

    if str(args.mode).upper() not in ['FULL', 'PATCH']:
        print('You must select a valid mode. Valid options are: FULL, PATCH.', file=sys.stderr)
        exit(-1)
    else:
        m = str(args.mode).upper()

    if str(args.color).upper() not in ['RGB', 'GRAY', 'HSV']:
        print('You must select a valid color space. Valid options are: RGB, GRAY, HSV.', file=sys.stderr)
        exit(-1)
    else:
        c = str(args.color).upper()

    if str(args.extra).upper() not in ['-', 'CANNY', 'PCA', 'WAV']:
        print('You must select a valid feature extraction. Valid options are: CANNY, PCA, WAV', file=sys.stderr)
        exit(-1)
    else:
        e = str(args.extra).upper()

    return f, a, m, c, e


if __name__ == '__main__':

    data, architecture, mode, color_space, feature_extracted = parse_input()

    name = '{}_{}'.format(mode, color_space) if feature_extracted == '-' else '{}_{}_{}'.format(mode, color_space, feature_extracted)

    print('Selected architecture : {}'.format(architecture))

    train_dataset = None
    validation_dataset = None
    test_dataset = None

    # Dataset initialization
    dataset = Dataset(folder=data, mode=mode, color_space=color_space, feature=feature_extracted)

    train, val, test = dataset.train_val_test_split()

    # Select loader

    original = None
    patch = None
    if feature_extracted == '-':
        original = dataset.load_data
        patch = dataset.load_patch_data
    elif feature_extracted == 'CANNY':
        original = dataset.load_data_canny
        patch = dataset.load_data_canny
    elif feature_extracted == 'PCA':
        original = dataset.load_data_pca
        patch = dataset.load_data_pca
    elif feature_extracted == 'WAV':
        original = dataset.load_data_wavelet
        patch = dataset.load_data_wavelet
    else:
        print('Feature extraction method not found.', file=sys.stderr)
        exit(-1)

    batch_size = 32

    if mode == 'FULL':
        # # Check dataset
        # dataset.random_plot()
        train_dataset = dataset.create_dataset(train, loader=original, batch_size=batch_size, shuffle=False)
        validation_dataset = dataset.create_dataset(val, loader=original, batch_size=batch_size, shuffle=False)
        test_dataset = dataset.create_dataset(test, loader=original, batch_size=batch_size, shuffle=False)
    elif mode == 'PATCH':
        # Patches
        batch_size = 32
        train_dataset = dataset.create_dataset(train, loader=patch, batch_size=batch_size, shuffle=False)
        validation_dataset = dataset.create_dataset(val, loader=patch, batch_size=batch_size, shuffle=False)
        test_dataset = dataset.create_dataset(test, loader=patch, batch_size=batch_size, shuffle=False)

    input_size = dataset.dim

    model = None

    # Check selected architecture
    if architecture == 'CNN':

        model = CNN(name=name, classes=3, shape=input_size, batch_size=batch_size)
        model.build()
        model.fit(train=train_dataset, validation=validation_dataset, num_epochs=20, steps=[len(train) // batch_size, len(val) // batch_size])
        model.save()
        # model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']], steps=len(test) // 32)

    if architecture == 'DNN':

        model = DNN(name=name, classes=3, shape=input_size, batch_size=batch_size)
        model.build(units=[512, 256, 128, 64])
        model.fit(train=train_dataset, validation=validation_dataset, num_epochs=20, steps=[len(train) // batch_size, len(val) // batch_size])
        model.save()
        # model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']], steps=len(test) // 32)

    elif architecture == 'D-CNN':

        model = DCNN(name=name, classes=3, shape=input_size, batch_size=batch_size)
        model.build()
        model.fit(train=train_dataset, validation=validation_dataset, num_epochs=20,
                  steps=[len(train) // batch_size, len(val) // batch_size])
        model.save()
        # model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']], steps=len(test) // 32)

    elif architecture == 'AE-DNN':

        model = AEDNN(name=name, classes=3, shape=input_size, batch_size=batch_size, code_size=512)





