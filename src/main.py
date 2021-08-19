import argparse
import sys

from Dataset import *
from models import CNN, DNN, DCNN, AEDNN


def parse_input():

    f = ''
    a = ''
    m = ''
    c = ''

    # Create parser
    parser = argparse.ArgumentParser(description='3-CLASS LYMPHOMA SUBTYPE CLASSIFICATION')

    # Add arguments
    parser.add_argument('-f', '--folder', type=str, default='data', help='dataset folder path')
    parser.add_argument('-a', '--arch', type=str, default='CNN', help='architecture, it can be CNN, DNN, D-CNN, AE-DNN')
    parser.add_argument('-m', '--mode', type=str, default='FULL', help='preprocessing mode, it can be FULL or PATCH')
    parser.add_argument('-c', '--color', type=str, default='RGB', help='color space used, it can be RGB, GRAY, HSV')

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

    return f, a, m, c


if __name__ == '__main__':

    data, architecture, mode, color_space = parse_input()

    print('Selected architecture : {}'.format(architecture))

    train_dataset = None
    validation_dataset = None
    test_dataset = None

    # Dataset initialization
    dataset = Dataset(folder=data, mode=mode, color_space=color_space)

    train, val, test = dataset.train_val_test_split()



    if mode == 'FULL':
        # # Check dataset
        # dataset.random_plot()
        train_dataset = dataset.create_dataset(train, loader=dataset.load_data, batch_size=32, shuffle=False)
        validation_dataset = dataset.create_dataset(val, loader=dataset.load_data, batch_size=32, shuffle=False)
        test_dataset = dataset.create_dataset(test, loader=dataset.load_data, batch_size=32, shuffle=False)
    elif mode == 'PATCH':
        # Patches
        train_dataset = dataset.create_dataset(train, loader=dataset.load_patches_data, batch_size=32, shuffle=False)
        validation_dataset = dataset.create_dataset(val, loader=dataset.load_patches_data, batch_size=32, shuffle=False)
        test_dataset = dataset.create_dataset(test, loader=dataset.load_patches_data, batch_size=32, shuffle=False)

    input_size = dataset.dim

    model = None

    # Check selected architecture
    if architecture == 'CNN':

        model = CNN(name='baseline', classes=3, shape=input_size, batch_size=32)

    if architecture == 'DNN':

        model = DNN(name='baseline', classes=3, shape=input_size, batch_size=32)

    elif architecture == 'D-CNN':

        model = DCNN(name='baseline', classes=3, shape=input_size, batch_size=32)

    elif architecture == 'AE-DNN':

        model = AEDNN(name='baseline', classes=3, shape=input_size, batch_size=32, code_size=512)

    model.build()
    model.fit(train=train_dataset, validation=validation_dataset, num_epochs=20, steps=[len(train) // 32, len(val) // 32])
    model.save()
    # model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']], steps=len(test) // 32)



