import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense, MaxPool2D

from .Network import NeuralNetwork


class Deep(NeuralNetwork):

    def __init__(self, name, classes, shape, batch_size=32):

        self._classes = classes
        self._shape = shape
        self._batch_size = batch_size

        self._name = 'deep_' + name
        self._output = 'output/{}.h5'.format(self._name)
        self._best = 'best/{}.h5'.format(self._name)

        self._loaded = False

        if os.path.exists(self._output):
            self._model = keras.models.load_model(self._output)
            self._loaded = True
        else:
            self._model = None

        self._history = None

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

    def fit(self, train, validation, num_epochs, steps: list):

        print("Model fit ...")

        # Compile model
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        # best_model_checkpoint = ModelCheckpoint(self._best, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.1, min_lr=0.000001, verbose=1)
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)

        self._history = self._model.fit(
            train,
            steps_per_epoch=steps[0],  # train steps
            epochs=num_epochs,
            validation_data=validation,
            validation_steps=steps[1],  # validation steps
            verbose=2,
            callbacks=[early_stop]
        )

    def save(self):

        print("Model save ...")

        self._model.save(self._output)
        print('Model saved!')

    def predict(self, test, labels):

        print("Model predict ...")

        # Extract labels of test set, predict them with the model
        prediction = self._model.predict(test)

        for pred in prediction:
            m = pred.max(axis=0)
            for i in range(pred.shape[0]):
                pred[i] = 1.0 if pred[i] == m else 0.0

        y_est_test = np.argmax(np.array(labels), axis=1)
        y_est_pred = np.argmax(np.array(prediction), axis=1)

        cm = confusion_matrix(y_est_test, y_est_pred)
        plot_confusion_matrix(cm=cm, classes=['CLL', 'FL', 'MCL'], path='output/cm_{}.png'.format(self._name), normalize=True)

        # Determine performance scores
        accuracy = accuracy_score(y_est_test, y_est_pred, normalize=True)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_est_test, y_est_pred, average='macro')

        print('PERFORMANCES ON TEST SET:')
        print('Accuracy: {:.2f}%'.format(accuracy * 100))
        print('Precision: {:.2f}%'.format(precision * 100))
        print('Recall: {:.2f}%'.format(recall * 100))
        print('Fscore: {:.2f}%'.format(fscore * 100))

    @property
    def is_loaded(self):
        return self._loaded


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          path='',
                          cmap=plt.cm.Blues):
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 4),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')

    plt.savefig(path)
    plt.show()

