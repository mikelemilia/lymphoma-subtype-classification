import argparse
import sys

from Dataset import *
from models import CNN, DNN, DCNN


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
    parser.add_argument('-e', '--extra', type=str, default='-',
                        help='extracted features, it can be CANNY, PCA, WAV, BLOB')

    # Retrieve arguments value
    args = parser.parse_args()

    # Check retrieved arguments
    if not os.path.exists(args.folder):
        print('You must provide an existing data location', file=sys.stderr)
        exit(-1)
    else:
        f = args.folder

    if str(args.arch).upper() not in ['CNN', 'DNN', 'D-CNN', 'AE-DNN', 'RNN']:
        print('You must select a valid implemented architecture. Valid options are: CNN, D-CNN, AE-DNN.',
              file=sys.stderr)
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

    if str(args.extra).upper() not in ['-', 'CANNY', 'PCA', 'WAV', 'BLOB']:
        print('You must select a valid feature extraction. Valid options are: CANNY, PCA, WAV, BLOB', file=sys.stderr)
        exit(-1)
    else:
        e = str(args.extra).upper()

    return f, a, m, c, e


if __name__ == '__main__':

    data, architecture, mode, color_space, feature_extracted = parse_input()

    name = '{}_{}'.format(mode, color_space) if feature_extracted == '-' else '{}_{}_{}'.format(mode, color_space,
                                                                                                feature_extracted)

    print('Selected architecture : {}'.format(architecture))

    train_dataset = None
    validation_dataset = None
    test_dataset = None

    # Dataset initialization
    dataset = Dataset(folder=data, mode=mode, color_space=color_space, feature=feature_extracted)

    dataset.generate_base_dataframe()
    train, val, test = dataset.train_val_test_split()

    # Select loader
    original = None
    patch = None
    data = None

    batch_size = 0
    patched = False

    if mode == 'FULL':
        batch_size = 32

        if feature_extracted == '-':
            original = dataset.load_data
        elif feature_extracted == 'CANNY':
            original = dataset.load_data_canny
        elif feature_extracted == 'BLOB':
            original = dataset.load_data_blob
        elif feature_extracted == 'PCA':
            original = dataset.load_data_pca
        elif feature_extracted == 'WAV':
            original = dataset.load_data_wavelet
        else:
            print('Feature extraction method not found.', file=sys.stderr)
            exit(-1)

        data = original(dataframe=train, index=0)

        # Augment datasets
        train = dataset.generate_augmented_dataframe(df=train, name='train')
        val = dataset.generate_augmented_dataframe(df=val, name='val')
        test = dataset.generate_augmented_dataframe(df=test, name='test')

        train_dataset = dataset.create_dataset(loader=original, dataframe=train, batch_size=batch_size, shuffle=False)
        validation_dataset = dataset.create_dataset(loader=original, dataframe=val, batch_size=batch_size, shuffle=False)
        test_dataset = dataset.create_dataset(loader=original, dataframe=test, batch_size=batch_size, shuffle=False)

    elif mode == 'PATCH':
        batch_size = 32
        patched = True

        if feature_extracted == '-':
            patch = dataset.load_patch_data
        elif feature_extracted == 'CANNY':
            patch = dataset.load_data_canny
        elif feature_extracted == 'BLOB':
            patch = dataset.load_data_blob
        elif feature_extracted == 'PCA':
            patch = dataset.load_data_pca
        elif feature_extracted == 'WAV':
            patch = dataset.load_data_wavelet
        else:
            print('Feature extraction method not found.', file=sys.stderr)
            exit(-1)

        data = patch(dataframe=train, index=0)

        # Patch datasets
        train = dataset.generate_patch_dataframe(df=train, name='train')
        val = dataset.generate_patch_dataframe(df=val, name='val')
        test = dataset.generate_patch_dataframe(df=test, name='test')

        train_dataset = dataset.create_dataset(dataframe=train, loader=patch, batch_size=batch_size, shuffle=False)
        validation_dataset = dataset.create_dataset(dataframe=val, loader=patch, batch_size=batch_size, shuffle=False)
        test_dataset = dataset.create_dataset(dataframe=test, loader=patch, batch_size=batch_size, shuffle=False)

    input_size = data.shape
    num_epochs = 50

    model = None

    # Check selected architecture
    if architecture == 'CNN':

        model = CNN(name=name, classes=3, shape=input_size, batch_size=batch_size, patched_image=patched)

        if not model.is_loaded:
            model.build()
            model.fit(train=train_dataset, validation=validation_dataset, num_epochs=num_epochs,
                      steps=[len(train) // batch_size, len(val) // batch_size])
            model.save()

        if patched:
            model.predict(dataframe=test, test=test_dataset, loader=patch)
        else:
            model.predict(dataframe=test, test=test_dataset, loader=original)

    if architecture == 'DNN':

        model = DNN(name=name, classes=3, shape=input_size, batch_size=batch_size)

        if not model.is_loaded:
            model.build(hidden_units=[1024, 512, 256, 128, 64])
            model.fit(train=train_dataset, validation=validation_dataset, num_epochs=num_epochs,
                      steps=[len(train) // batch_size, len(val) // batch_size])
            model.save()

        model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']])

    elif architecture == 'D-CNN':

        model = DCNN(name=name, classes=3, shape=input_size, batch_size=batch_size)

        if not model.is_loaded:
            model.build()
            model.fit(train=train_dataset, validation=validation_dataset, num_epochs=num_epochs,
                      steps=[len(train) // batch_size, len(val) // batch_size])
            model.save()

        model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']])

    elif architecture == 'AE-DNN':
        pass
        # model = AEDNN(name=name, classes=3, shape=input_size, batch_size=batch_size, code_size=512)
        #
        # if not model.is_loaded:
        #     model.build()
        #     model.fit(train=train_dataset, validation=validation_dataset, num_epochs=num_epochs,
        #               steps=[len(train) // batch_size, len(val) // batch_size])
        #     model.save()
        #
        # model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']])
