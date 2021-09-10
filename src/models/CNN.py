from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from .Network import NeuralNetwork


class Convolutional(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):

        super().__init__('CNN_' + name, classes, shape, batch_size, patched_image)

    def build(self, hidden_units: list):

        print("Model build ...")

        x_input = Input(self._shape)

        # Layer with 32x32 Conv2D
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu', padding='same', name='C32')(x_input)
        x = MaxPooling2D(pool_size=2, name='MP32')(x)
        x = Dropout(rate=0.25, name='D32')(x)

        # Layer with 64x64 Conv2D
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu', padding='same', name='C64')(x)
        x = MaxPooling2D(pool_size=2, name='MP64')(x)
        x = Dropout(rate=0.25, name='D64')(x)

        x = Flatten()(x)

        for i, units in enumerate(hidden_units):
            x = Dense(units=units, activation='relu', name='FC{}'.format(units))(x)
            x = Dropout(rate=0.25, name='D{}'.format(units))(x)

        x = Dense(self._classes, activation='softmax', name='FC')(x)

        self._model = Model(inputs=x_input, outputs=x, name=self._name)

        # Check model
        self._model.summary()
