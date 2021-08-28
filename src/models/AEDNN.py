import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, MaxPool2D, Reshape
from tensorflow.keras.regularizers import l1

from .Network import NeuralNetwork


class AutoEncoder(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, code_size=512, patched_image: bool = False):

        self._classes = classes
        self._shape = shape
        self._batch_size = batch_size
        self._code_size = code_size

        self._patched_image = patched_image

        self._name = 'ae_' + name
        self._output = 'output/{}.h5'.format(self._name)

        self._name_deep = 'ae_deep_' + name
        self._output_deep = 'output/{}.h5'.format(self._name_deep)

        self._loaded = False

        if os.path.exists(self._output):
            self._model = keras.models.load_model(self._output)
            self._loaded = True
        else:
            self._model = None

        self._encoder = None
        self._decoder = None

        self._deep = False

        self._history = None

    def build(self):

        if self._patched_image:
            mid_shape = (16, 16, 128)
        else:
            mid_shape = (26, 35, 128)

        # Encoder
        encoder_input = Input(self._shape, name='encoder_input')

        encoder = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='conv64')(encoder_input)
        encoder = MaxPool2D((2, 2), padding='same')(encoder)
        encoder = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='conv128')(encoder)
        encoder = MaxPool2D((2, 2), padding='same')(encoder)

        encoder = Flatten()(encoder)
        encoder = Dense(units=self._code_size)(encoder)

        self._encoder = Model(inputs=encoder_input, outputs=encoder, name='encoder')
        self._encoder.summary()

        # Decoder
        decoder_input = Input(self._code_size, name='decoder_input')

        decoder = Dense(units=np.prod(mid_shape), activation='relu')(decoder_input)
        decoder = Reshape(mid_shape)(decoder)
        decoder = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same', name='convT128')(decoder)
        decoder = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same', name='convT64')(decoder)
        decoder = Conv2DTranspose(filters=self._shape[2], kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same', name='convT')(decoder)

        self._decoder = Model(inputs=decoder_input, outputs=decoder, name='decoder')
        self._decoder.summary()

        # Auto-encoder model
        ae_input = Input(self._shape, name='input')
        code = self._encoder(ae_input)
        ae_output = self._decoder(code)

        self._model = Model(inputs=ae_input, outputs=ae_output, name=self._name)

        # Check model
        self._model.summary()

    def stack_deep(self, hidden_units):

        if self._model is None and os.path.exists(self._output):
            self._model = keras.models.load_model(self._output)
            self._deep = True
        else:
            print('Unable to locate the saved .h5 model', file=sys.stderr)
            exit(-1)

        model = Sequential()

        for layer in self._model.layers[:-1]:
            model.add(layer)

        for layer in model.layers:
            layer.trainable = False

        for i, units in enumerate(hidden_units):
            model.add(Dense(units=units, activation='relu', name='fc{}'.format(i), kernel_regularizer=l1(0.0001)))

        model.add(Dense(units=self._classes, activation='softmax', name='fc'))
        model.build()

        self._model = model

        # Check model
        self._model.summary()

    def fit(self, train, validation, num_epochs, steps: list):

        # Compile model
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        # best_model_checkpoint = ModelCheckpoint(self._best, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)

        self._history = self._model.fit(
            train,
            steps_per_epoch=steps[0],  # train steps
            epochs=num_epochs,
            validation_data=validation,
            validation_steps=steps[1],  # validation steps
            verbose=1,
            callbacks=[early_stop]
        )

    def save(self):

        if self._deep:
            self._model.save(self._output_deep)
        else:
            self._model.save(self._output)

        print('Model saved!')

    def predict(self, dataframe, test, loader):

        print("Model predict ...")

        def one_hot_encode(preds):

            for pred in preds:
                val = pred.max(axis=0)
                for k in range(3):  # 3-class classification
                    pred[k] = 1.0 if pred[k] == val else 0.0

            return preds

        if self._patched_image:  # input composed of patches

            print("Decision fusion mechanism ...")

            images = dataframe.drop_duplicates(subset=['path'])
            print('Images found : {}'.format(len(images)))
            images_pred = np.zeros((images.shape[0], 3))

            print('Computing frequency prediction for each image')
            count = 0
            for i, image in images.iterrows():

                print('\tImage {} : {}'.format(count, image['path']))
                patches = dataframe.loc[dataframe['path'] == image['path']]
                print('\t\tPatch found : {}'.format(len(patches)))

                patches_pred = np.zeros((patches.shape[0], 3))

                for j, _ in patches.iterrows():
                    patch = loader(split=2, index=j)

                    patch = self._model.predict(patch)
                    patches_pred[j, :] = one_hot_encode(patch)

                prediction, frequency = np.unique(patches_pred, axis=0, return_counts=True)
                print('\t\tPrediction :')
                for h in range(prediction.shape[0]):
                    print('\t\t\tClass {} : {}'.format(prediction[h], frequency[h]))

                images_pred[count, :] = prediction[np.argmax(frequency)]
                count += 1

        else:  # input is directly full image

            # Extract labels of test set, predict them with the model
            images_pred = self._model.predict(test)
            images_pred = one_hot_encode(images_pred)

        # Extract labels from dataframe
        labels = dataframe[['label_cll', 'label_fl', 'label_mcl']]

        y_est_test = np.argmax(np.array(labels), axis=1)
        y_est_pred = np.argmax(np.array(images_pred), axis=1)

        cm = confusion_matrix(y_est_test, y_est_pred)
        plot_confusion_matrix(cm=cm, classes=['CLL', 'FL', 'MCL'], path='output/cm_{}.png'.format(self._name),
                              normalize=False)

        # Determine performance scores
        accuracy = accuracy_score(y_est_test, y_est_pred, normalize=True)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_est_test, y_est_pred, average='macro')

        print('PERFORMANCES ON TEST SET:')
        print('Accuracy: {:.2f}%'.format(accuracy * 100))
        print('Precision: {:.2f}%'.format(precision * 100))
        print('Recall: {:.2f}%'.format(recall * 100))
        print('Fscore: {:.2f}%'.format(fscore * 100))

    @property
    def is_loaded(self):
        return self._loaded


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          path='',
                          cmap=plt.cm.Blues):
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 4),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')

    plt.savefig(path)
    plt.show()
