import os
import sys

import numpy as np
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, MaxPool2D, Reshape
from tensorflow.keras.regularizers import l1

from .Network import NeuralNetwork


class AutoEncoder(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, code_size=512, patched_image: bool = False):

        super().__init__('ae_' + name, classes, shape, batch_size, patched_image)

        self._code_size = code_size

        self._name_deep = 'ae_deep_' + name
        self._output_deep = 'output/{}.h5'.format(self._name_deep)

        self._encoder = None
        self._decoder = None

        self._deep = False

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
