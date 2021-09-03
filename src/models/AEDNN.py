import os
import sys

import numpy as np
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, UpSampling2D, Reshape, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2

from .Network import NeuralNetwork


class AutoEncoder(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, code_size=512, patched_image: bool = False):

        super().__init__('AE_' + name, classes, shape, batch_size, patched_image)

        self._code_size = code_size

        self._name_deep = 'AEDNN_' + name
        self._output_deep = 'output/{}.h5'.format(self._name_deep)

        self._encoder = None
        self._decoder = None

        self._deep = False

    def build(self):

        mid_shape = (None, None, None)

        if self._patched_image:
            mid_shape = (32, 32, 128)

        # Encoder
        encoder_input = Input(self._shape, name='encoder_input')

        # encoder = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
        #                  data_format='channels_last', use_bias=True,
        #                  activation='elu',
        #                  padding='same',
        #                  name='conv32')(encoder_input)
        #
        # encoder = MaxPooling2D(pool_size=2, padding='same')(encoder)

        encoder = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                         data_format='channels_last', use_bias=True,
                         activation='elu',
                         padding='same',
                         name='conv64')(encoder_input)

        encoder = MaxPooling2D(pool_size=2, padding='same')(encoder)

        encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                         data_format='channels_last', use_bias=True,
                         activation='elu',
                         padding='same',
                         name='conv128')(encoder)

        encoder = MaxPooling2D(pool_size=2, padding='same')(encoder)

        encoder = Flatten()(encoder)
        encoder = Dense(units=self._code_size)(encoder)

        self._encoder = Model(inputs=encoder_input, outputs=encoder, name='encoder')
        # self._encoder.summary()

        # Decoder
        decoder_input = Input(self._code_size, name='decoder_input')

        decoder = Dense(np.prod(mid_shape), activation='relu')(decoder_input)
        decoder = Reshape(mid_shape)(decoder)
        decoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                         data_format='channels_last', use_bias=True,
                         activation='elu',
                         padding='same',
                         name='conv128')(decoder)

        decoder = UpSampling2D(size=2)(decoder)

        decoder = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                         data_format='channels_last', use_bias=True,
                         activation='elu',
                         padding='same',
                         name='conv64')(decoder)

        decoder = UpSampling2D(size=2)(decoder)

        # decoder = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
        #                  data_format='channels_last', use_bias=True,
        #                  activation='elu',
        #                  padding='same',
        #                  name='conv32')(decoder)
        # decoder = UpSampling2D(size=2)(decoder)

        self._decoder = Model(inputs=decoder_input, outputs=decoder, name='decoder')
        # self._decoder.summary()

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

        regularizer = l2(0.0001)
        for i, units in enumerate(hidden_units):
            model.add(Dense(units=units, activation='relu', name='fc{}'.format(i), kernel_regularizer=regularizer))

        model.add(Dropout(rate=0.3))

        model.add(Dense(units=self._classes, activation='softmax', name='fc'))
        model.build()

        self._model = model

        # Check model
        self._model.summary()
