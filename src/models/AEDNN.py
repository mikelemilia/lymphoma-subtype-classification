import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, MaxPool2D

from .Network import NeuralNetwork


class AutoEncoderDeep(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, code_size=512):

        self._classes = classes
        self._shape = shape
        self._batch_size = batch_size
        self._code_size = code_size

        self._name = 'ae_deep_' + name
        self._output = 'output/models/{}.h5'.format(self._name)
        self._best = 'best/models/{}.h5'.format(self._name)

        self._model = None
        self._encoder = None
        self._decoder = None

        self._history = None

    def build(self):

        # Encoder

        encoder_input = Input(self._shape, name='encoder_input')

        encoder = Conv2D(ilters=64, kernel_size=(3, 3), activation='elu', padding='same', name='conv64')(encoder_input)
        encoder = MaxPool2D((1, 2), padding='same')(encoder)
        encoder = Conv2D(ilters=128, kernel_size=(3, 3), activation='elu', padding='same', name='conv128')(encoder)
        encoder = MaxPool2D((2, 2), padding='same')(encoder)

        encoder = Flatten()(encoder)
        encoder = Dense(units=self._code_size)(encoder)

        self._encoder = Model(inputs=encoder_input, outputs=encoder, name=self._name + '_encoder')

        # Decoder

        decoder_input = Input(self._code_size, name='decoder_input')

        decoder = Dense(units=np.prod(self._shape), activation='elu')(decoder_input)
        decoder = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same', name='convT64')(decoder)
        decoder = Conv2DTranspose(filters=self._classes, kernel_size=(3, 3), strides=2, activation='elu', padding='same', name='convT')(decoder)

        self._decoder = Model(inputs=decoder_input, outputs=decoder, name=self._name + '_decoder')

        # Auto-encoder model

        self._model = Model(inputs=self._encoder, output=self._decoder, name=self._name)

        # Check model
        self._model.summary()

    def stack(self):

        x_input = Input(self._shape, name='input')

        for layer in self._encoder.layers:
            layer.trainable = False

        # TODO : implement DNN

    def fit(self, train, validation, num_epochs, steps: list):

        # Compile model
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        best_model_checkpoint = ModelCheckpoint(self._best, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.1, min_lr=0.000001, verbose=1)
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)

        self._history = self._model.fit(
            train,
            steps_per_epoch=steps[0],  # train steps
            epochs=num_epochs,
            validation_data=validation,
            validation_steps=steps[1],  # validation steps
            verbose=1,
            callbacks=[best_model_checkpoint, reduce_lr, early_stop]
        )

    def save(self):

        self._model.save(self._output)
        print('Model saved!')

    def predict(self, test, labels, steps):

        # Load saved model
        if self._model is None and os.path.exists(self._output):
            self._model = keras.models.load_model(self._output)
        else:
            print('Unable to locate the saved .h5 model', file=sys.stderr)
            exit(-1)

        # Extract labels of test set, predict them with the model
        prediction = self._model.predict(test, steps=steps)
        test_est_classes = (prediction > 0.5).astype(int)

        # Determine performance scores
        accuracy = accuracy_score(labels, test_est_classes, normalize=True)
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, test_est_classes, average='macro')

        print('PERFORMANCES ON TEST SET:')
        print('Accuracy: {:.2f}%'.format(accuracy * 100))
        print('Precision: {:.2f}%'.format(precision * 100))
        print('Recall: {:.2f}%'.format(recall * 100))
        print('Fscore: {:.2f}%'.format(fscore * 100))

        # Plot of loss-accuracy and ROC

        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Loss, accuracy and ROC')
        # Plot loss
        axs[0, 0].plot(self._history.history['loss'], label='Train loss')
        axs[0, 0].plot(self._history.history['val_loss'], label='Val loss')
        axs[0, 0].legend()
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_title('Loss')
        # Plot accuracy
        axs[1, 0].plot(self._history.history['accuracy'], label='Train accuracy')
        axs[1, 0].plot(self._history.history['val_accuracy'], label='Val accuracy')
        axs[1, 0].legend()
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].set_title('Accuracy')

        # if len(label) == 1:
        #     fpr, tpr, _ = roc_curve(test_labels, test_est_classes)
        #     roc_auc = auc(fpr, tpr)
        #     # Plot ROC when only 1 label is present
        #     axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        #     axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #     axs[0, 1].set_xlabel('False Positive Rate')
        #     axs[0, 1].set_ylabel('True Positive Rate')
        #     axs[0, 1].set_title('ROC for {}'.format(label))
        # else:
        #     for l in range(len(label)):
        #         fpr, tpr, _ = roc_curve(test_labels[:, l], test_est_classes[:, l])
        #         roc_auc = auc(fpr, tpr)
        #         # Plot ROC for each of the two labels
        #         axs[l, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        #         axs[l, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #         axs[l, 1].set_xlabel('False Positive Rate')
        #         axs[l, 1].set_ylabel('True Positive Rate')
        #         axs[l, 1].set_title('ROC for {}'.format(label[l]))
        # plt.savefig('images/{}_evaluation'.format(title))
