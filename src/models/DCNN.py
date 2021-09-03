from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, AveragePooling2D

from .Network import NeuralNetwork


class DeepConvolutional(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):

        super().__init__('DCNN_' + name, classes, shape, batch_size, patched_image)

    def build(self):

        print("Model build ...")

        x_input = Input(self._shape, name='input')

        # Layer with 64x64 Conv2D
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same',
                   name='conv64_A')(x_input)

        # Layer with 64x64 Conv2D
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   data_format='channels_last', use_bias=True,
                   activation='relu',
                   padding='same',
                   name='conv64_B')(x)

        x = AveragePooling2D(pool_size=2)(x)
        # x = Dropout(rate=0.15, name='dropout_15_A')(x)

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

        x = AveragePooling2D(pool_size=2)(x)
        # x = Dropout(rate=0.15, name='dropout_15_B')(x)

        # # Layer with 256x256 Conv2D
        # x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
        #            data_format='channels_last', use_bias=True,
        #            activation='relu',
        #            padding='same',
        #            name='conv256_A')(x)
        #
        # # Layer with 256x256 Conv2D
        # x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
        #            data_format='channels_last', use_bias=True,
        #            activation='relu',
        #            padding='same',
        #            name='conv256_B')(x)
        #
        # x = AveragePooling2D(pool_size=2)(x)
        # x = Dropout(rate=0.15, name='dropout_15_C')(x)

        x = Flatten()(x)

        # x = Dense(512, activation='relu', name='fc512')(x)
        x = Dense(256, activation='relu', name='fc256')(x)
        # x = Dense(128, activation='relu', name='fc128')(x)
        x = Dropout(rate=0.3, name='dropout_30')(x)

        x = Dense(self._classes, activation='softmax', name='fc')(x)

        self._model = Model(inputs=x_input, outputs=x, name=self._name)

        # Check model
        self._model.summary()
