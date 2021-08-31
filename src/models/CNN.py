from .Network import NeuralNetwork

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, MaxPool2D, BatchNormalization


class Convolutional(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):

        super().__init__('cnn_' + name, classes, shape, batch_size, patched_image)

    def build(self):

        print("Model build ...")

        x_input = Input(self._shape, name='input')

        # Layer with 64x64 Conv2D
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', use_bias=True, padding='same', name='conv64')(x_input)
        # x = BatchNormalization(axis=-1, name='bn64')(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.25)(x)

        # Layer with 128x128 Conv2D
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', use_bias=True, padding='same', name='conv128')(x)
        # x = BatchNormalization(axis=-1, name='bn128')(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.25)(x)

        x = MaxPool2D(pool_size=2)(x)

        x = Flatten()(x)
        x = Dense(128, name='fc128')(x)
        x = Activation('elu')(x)
        x = Dropout(rate=0.25)(x)
        x = Dense(64, name='fc64')(x)
        x = Activation('elu')(x)
        x = Dropout(rate=0.25)(x)

        x = Dense(self._classes, activation='softmax', name='fc')(x)

        self._model = Model(inputs=x_input, outputs=x, name=self._name)

        # Check model
        self._model.summary()