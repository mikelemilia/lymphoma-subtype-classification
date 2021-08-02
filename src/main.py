import sys
import os

from Dataset import *

from models import CNN, DNN, DCNN, AEDNN
import argparse


def parse_input():

    folder = ''
    arch = ''
    mode = ''
    patches = 0

    # Create parser
    parser = argparse.ArgumentParser(description='3-CLASS LYMPHOMA SUBTYPE CLASSIFICATION')

    # Add arguments
    parser.add_argument('-f', '--folder', type=str, default='data', help='dataset folder path')
    parser.add_argument('-a', '--arch', type=str, default='CNN', help='NN architecture, it can be CNN, DNN, D-CNN, AE-DNN, RNN')
    parser.add_argument('-m', '--mode', type=str, default='FULL', help='NN preprocessing mode, it can be FULL or PATCHED')
    parser.add_argument('-p', '--patch', type=int, help='number of patches')

    # Retrieve arguments value
    args = parser.parse_args()

    # Check retrieved arguments
    if not os.path.exists(args.folder):
        print('You must provide an existing data location', file=sys.stderr)
        exit(-1)
    else:
        folder = args.folder

    if str(args.arch).upper() not in ['CNN', 'DNN', 'D-CNN', 'AE-DNN', 'RNN']:
        print('You must select a valid implemented architecture! Valid options are: CNN, D-CNN, AE-DNN, RNN.', file=sys.stderr)
        exit(-1)
    else:
        arch = str(args.arch).upper()

    if str(args.mode).upper() not in ['FULL', 'PATCHED']:
        print('You must select a valid technique. Valid options are: FULL, PATCHED.', file=sys.stderr)
        exit(-1)
    else:
        mode = str(args.mode).upper()

    if mode in ['PATCHED']:
        if args.patches > 1:
            patches = int(args.patches)
        else:
            print('You must provide a number of patches at least greater or equal than 2.', file=sys.stderr)
            exit(-1)

    return folder, arch, mode, patches


if __name__ == '__main__':

    data, architecture, mode, patches = parse_input()

    # Dataset initialization
    dataset = Dataset(folder=data, mode=mode)

    # Check dataset
    dataset.random_plot()

    train, val, test = dataset.train_val_test_split()
    input_size = dataset.dim

    print('Selected architecture : {}'.format(architecture))

    train_dataset = dataset.create_dataset(train, loader=dataset.load_data, batch_size=32, shuffle=False)
    validation_dataset = dataset.create_dataset(val, loader=dataset.load_data, batch_size=32, shuffle=False)
    test_dataset = dataset.create_dataset(test, loader=dataset.load_data, batch_size=32, shuffle=False)

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
    model.predict(test=test_dataset, labels=test[['label_cll', 'label_fl', 'label_mcl']], steps=len(test) // 32)



