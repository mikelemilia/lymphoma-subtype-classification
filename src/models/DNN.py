from .Network import NeuralNetwork

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense


class Deep(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):

        super().__init__('deep_' + name, classes, shape, batch_size, patched_image)

    def build(self, hidden_units):

        print("Model build ...")

        x_input = Input(self._shape, name='input')

        x = Flatten()(x_input)

        for i, units in enumerate(hidden_units):
            x = Dense(units=units, activation='relu', name='fc{}'.format(i))(x)

        x = Dense(units=self._classes, activation='softmax', name='fc')(x)

        self._model = Model(inputs=x_input, outputs=x, name=self._name)

        # Check model
        self._model.summary()
