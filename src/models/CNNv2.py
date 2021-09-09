from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from .Network import NeuralNetwork


class CNNv2(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):

        super().__init__('CNNv2_' + name, classes, shape, batch_size, patched_image)

    def build(self, hidden_units: list):

        print("Model build ...")

        x_input = Input(self._shape)

        # Layer with 32x32 Conv2D
        a = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu', padding='same', name='C32')(x_input)
        a = MaxPooling2D(pool_size=2, name='MP32')(a)
        a = Dropout(rate=0.25, name='D32')(a)

        # Layer with 64x64 Conv2D
        b = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu', padding='same', name='C64')(a)
        b = MaxPooling2D(pool_size=2, name='MP64')(b)
        b = Dropout(rate=0.25, name='D64')(b)

        b = self.shortcut(input_block=a, residual_block=b)

        # Layer with 128x128 Conv2D
        c = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu', padding='same', name='C128')(b)
        c = MaxPooling2D(pool_size=2, name='MP128')(c)
        c = Dropout(rate=0.25, name='D128')(c)

        c = self.shortcut(input_block=b, residual_block=c)

        x = Flatten()(c)

        for i, units in enumerate(hidden_units):
            x = Dense(units=units, activation='relu', name='FC{}'.format(units))(x)
            x = Dropout(rate=0.25, name='D{}'.format(units))(x)

        x = Dense(self._classes, activation='softmax', name='FC')(x)

        self._model = Model(inputs=x_input, outputs=x, name=self._name)

        # Check model
        self._model.summary()

    def fit(self, train, validation, num_epochs, steps: list, patience_lr=5, patience_es=10):

        super().fit(train, validation, num_epochs, steps, patience_lr=10, patience_es=20)
