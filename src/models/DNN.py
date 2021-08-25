import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
            verbose=1,
            callbacks=[early_stop]
        )

    def save(self):

        print("Model save ...")

        self._model.save(self._output)
        print('Model saved!')

    def predict(self, test, labels, steps):

        print("Model predict ...")

        # Extract labels of test set, predict them with the model
        prediction = self._model.predict(test)
        test_est_classes = (prediction > 0.5).astype(int)

        # Determine performance scores
        accuracy = accuracy_score(labels, test_est_classes, normalize=True)
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, test_est_classes, average='macro')

        print('PERFORMANCES ON TEST SET:')
        print('Accuracy: {:.2f}%'.format(accuracy * 100))
        print('Precision: {:.2f}%'.format(precision * 100))
        print('Recall: {:.2f}%'.format(recall * 100))
        print('Fscore: {:.2f}%'.format(fscore * 100))

        # Plot of loss-accuracy and ROC

        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Loss, accuracy and ROC')
        # Plot loss
        axs[0, 0].plot(self._history.history['loss'], label='Train loss')
        axs[0, 0].plot(self._history.history['val_loss'], label='Val loss')
        axs[0, 0].legend()
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_title('Loss')
        # Plot accuracy
        axs[1, 0].plot(self._history.history['accuracy'], label='Train accuracy')
        axs[1, 0].plot(self._history.history['val_accuracy'], label='Val accuracy')
        axs[1, 0].legend()
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].set_title('Accuracy')

        # if len(label) == 1:
        #     fpr, tpr, _ = roc_curve(test_labels, test_est_classes)
        #     roc_auc = auc(fpr, tpr)
        #     # Plot ROC when only 1 label is present
        #     axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        #     axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #     axs[0, 1].set_xlabel('False Positive Rate')
        #     axs[0, 1].set_ylabel('True Positive Rate')
        #     axs[0, 1].set_title('ROC for {}'.format(label))
        # else:
        #     for l in range(len(label)):
        #         fpr, tpr, _ = roc_curve(test_labels[:, l], test_est_classes[:, l])
        #         roc_auc = auc(fpr, tpr)
        #         # Plot ROC for each of the two labels
        #         axs[l, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        #         axs[l, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #         axs[l, 1].set_xlabel('False Positive Rate')
        #         axs[l, 1].set_ylabel('True Positive Rate')
        #         axs[l, 1].set_title('ROC for {}'.format(label[l]))
        # plt.savefig('images/{}_evaluation'.format(title))

    @property
    def is_loaded(self):
        return self._loaded
