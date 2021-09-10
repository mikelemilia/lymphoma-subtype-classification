import argparse
import sys

from Dataset import *
from models import CNN, CNNv1, CNNv2, CNNv3, CNNv4


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
    parser.add_argument('-s', '--size', type=int, default=128, help='preprocessing mode, it can be 128, 64, 32')
    parser.add_argument('-c', '--color', type=str, default='RGB', help='color space used, it can be RGB, GRAY, HSV')
    parser.add_argument('-e', '--extra', type=str, default='-', help='extracted features, it can be CANNY, PCA, WAV, BLOB')
    parser.add_argument('--train', action='store_true')

    # Retrieve arguments value
    args = parser.parse_args()

    # Check retrieved arguments
    if not os.path.exists(args.folder):
        print('You must provide an existing data location', file=sys.stderr)
        exit(-1)
    else:
        f = args.folder

    if args.arch not in ['CNN', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4', 'CNNv5', 'CNNv6', 'CNNv7']:
        print('You must select a valid implemented architecture. Valid options are: CNN, D-CNN, AE-DNN.',
              file=sys.stderr)
        exit(-1)
    else:
        a = args.arch

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
        print('You must select a valid feature extraction. Valid options are: CANNY, PCA, WAV.', file=sys.stderr)
        exit(-1)
    else:
        e = str(args.extra).upper()

    t = args.train
    s = args.size

    return f, a, m, c, e, s, t


if __name__ == '__main__':

    data, architecture, mode, color_space, feature_extracted, patch_size, is_training = parse_input()

    name = '{}_{}'.format(mode, color_space) if feature_extracted == '-' else '{}_{}_{}'.format(mode, color_space, feature_extracted)

    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Dataset initialization
    dataset = Dataset(folder=data, mode=mode, color_space=color_space, feature=feature_extracted, is_training=is_training, patch_size=patch_size)
    # print('Selected architecture : {}'.format(architecture))

    dataset.generate_base_dataframe()
    train, val, test = dataset.split_dataset()

    # print('\nStarting training set : {} images'.format(len(train)))
    # print('Starting validation set  : {} images'.format(len(val)))
    # print('Starting test set  : {} images'.format(len(test)))

    # Select loader
    loader = dataset.select_loader()

    data = None
    patched = False
    batch_size = None
    num_epochs = 250

    if mode == 'FULL':

        # Augment datasets
        train = dataset.generate_augmented_dataframe(split=0, name='train')
        val = dataset.generate_augmented_dataframe(split=1, name='val')
        test = dataset.generate_augmented_dataframe(split=2, name='test')

        batch_size = 32

        # print('\nAugmented training set : {} images'.format(len(train)))
        # print('Augmented validation set  : {} images'.format(len(val)))
        # print('Augmented test set  : {} images'.format(len(test)))

    elif mode == 'PATCH':

        # Patch datasets
        train = dataset.generate_patch_dataframe(split=0, name='train')
        val = dataset.generate_patch_dataframe(split=1, name='val')
        test = dataset.generate_patch_dataframe(split=2, name='test')

        patched = True
        batch_size = 64

        # print('\nPatched training set : {} images'.format(len(train)))
        # print('Patched validation set  : {} images'.format(len(val)))
        # print('Patched test set  : {} images'.format(len(test)))

    train_dataset = dataset.create_dataset(loader=loader, dataframe=train, batch_size=batch_size, shuffle=False, split=0)
    val_dataset = dataset.create_dataset(loader=loader, dataframe=val, batch_size=batch_size, shuffle=False, split=1)
    test_dataset = dataset.create_dataset(loader=loader, dataframe=test, batch_size=batch_size, shuffle=False, split=2)

    data = loader(split=0, index=0)
    input_size = data.shape

    model = None

    # Check selected architecture
    if architecture in ['CNN', 'CNNv1', 'CNNv2', 'CNNv3', 'CNNv4', 'CNNv5', 'CNNv6', 'CNNv7']:

        model = globals()[architecture](name=name, classes=3, shape=input_size, batch_size=batch_size, patched_image=patched)

        if model.is_loaded:
            model.predict(dataframe=test, test=test_dataset, loader=loader)

        else:
            model.build(hidden_units=[256])
            model.fit(train=train_dataset, validation=val_dataset, num_epochs=num_epochs,
                      steps=[len(train) // batch_size, len(val) // batch_size])
            model.save()