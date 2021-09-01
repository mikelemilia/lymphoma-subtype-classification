from .Network import NeuralNetwork

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, MaxPool2D
from tensorflow.keras.regularizers import l2


class DeepConvolutional(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):

        super().__init__('deep_cnn_' + name, classes, shape, batch_size, patched_image)

    def build(self):

        print("Model build ...")

        x_input = Input(self._shape, name='input')

        # Layer with 64x64 Conv2D
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same', name='conv64_A')(x_input)

        # Layer with 64x64 Conv2D
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same',
                   name='conv64_B')(x)

        # x = MaxPool2D(pool_size=2)(x)
        x = Dropout(rate=0.25)(x)

        # Layer with 128x128 Conv2D
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same',
                   name='conv128_A')(x)

        # Layer with 128x128 Conv2D
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same',
                   name='conv128_B')(x)

        # x = MaxPool2D(pool_size=2)(x)
        x = Dropout(rate=0.25)(x)

        # Layer with 256x256 Conv2D
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same',
                   name='conv256_A')(x)

        # Layer with 128x128 Conv2D
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same',
                   name='conv256_B')(x)

        x = MaxPool2D(pool_size=2)(x)
        x = Dropout(rate=0.25)(x)

        x = Flatten()(x)

        x = Dense(512, name='fc512', activation='relu')(x)

        x = Dropout(rate=0.5)(x)

        x = Dense(self._classes, activation='softmax', name='fc')(x)

        self._model = Model(inputs=x_input, outputs=x, name=self._name)

        # Check model
        self._model.summary()
