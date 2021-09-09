from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

from .Network import NeuralNetwork


class CNNv4(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):

        super().__init__('CNNv4_' + name, classes, shape, batch_size, patched_image)

    def build(self, hidden_units: list):

        print("Model build ...")

        x_input = Input(self._shape)

        # Layer with 32x32 Conv2D
        a = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu',
                   padding='same', name='C32')(x_input)
        a = MaxPooling2D(pool_size=2, name='MP32')(a)
        a = Dropout(rate=0.25, name='D32')(a)

        # Layer with 64x64 Conv2D
        b = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu',
                   padding='same', name='C64')(a)
        b = MaxPooling2D(pool_size=2, name='MP64')(b)
        b = Dropout(rate=0.25, name='D64')(b)

        b = self.shortcut(a, b)

        # Layer with 128x128 Conv2D
        c = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', activation='relu',
                   padding='same', name='C128')(b)
        c = MaxPooling2D(pool_size=2, name='MP128')(c)
        c = Dropout(rate=0.25, name='D128')(c)

        c = self.shortcut(b, c)

        x = GlobalAveragePooling2D(data_format='channels_last')(c)
        x = Dropout(rate=0.25, name='D')(x)

        x = Dense(self._classes, activation='softmax', name='FC')(x)

        self._model = Model(inputs=x_input, outputs=x, name=self._name)

        # Check model
        self._model.summary()

    def fit(self, train, validation, num_epochs, steps: list, patience_lr=5, patience_es=10):
        print("Model fit ...")

        # Compile model
        optimizer = SGD(learning_rate=0.01, momentum=0.01, nesterov=True)
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        patience_lr = 10
        patience_es = 20

        # Callbacks definition
        reduce_lr = ReduceLROnPlateau(
            monitor='val_categorical_accuracy', mode='max',
            factor=0.1, min_lr=0.0000001, patience=patience_lr, verbose=1)

        early_stop = EarlyStopping(
            monitor='val_categorical_accuracy', mode='max',
            patience=patience_es, verbose=1, restore_best_weights=True)

        # Fit model
        self._history = self._model.fit(
            train,
            steps_per_epoch=steps[0],  # training steps
            epochs=num_epochs,
            validation_data=validation,
            validation_steps=steps[1],  # validation steps
            verbose=2,
            callbacks=[
                reduce_lr,
                early_stop
            ]
        )

        # Plot loss and accuracy curves
        self.plot_curves()
