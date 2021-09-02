import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


class NeuralNetwork:

    def __init__(self, name, classes, shape, batch_size=32, patched_image: bool = False):
        self._classes = classes
        self._shape = shape
        self._batch_size = batch_size

        self._patched_image = patched_image

        self._name = name
        self._output = 'output/{}.h5'.format(self._name)

        self._loaded = False

        if os.path.exists(self._output):
            self._model = keras.models.load_model(self._output)
            self._loaded = True
        else:
            self._model = None

        self._history = None

    def build(self, *args):
        pass

    def fit(self, train, validation, num_epochs, steps: list):

        print("Model fit ...")

        # Compile model
        optimizer = Adam(learning_rate=0.001)
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Callbacks definition
        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max',
                                      patience=5, factor=0.1, min_lr=0.00001,
                                      verbose=1)

        early_stop = EarlyStopping(monitor='val_categorical_accuracy', mode='max',
                                   patience=10,
                                   verbose=1)

        # Fit model
        self._history = self._model.fit(
            train,
            steps_per_epoch=steps[0],   # training steps
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

    def save(self):

        print("Model save ...")

        self._model.save(self._output)
        print('Model saved!')

    def predict(self, dataframe, test, loader):

        print("Model predict ...")
        labels = None

        def one_hot_encode(preds):

            for pred in preds:
                val = pred.max(axis=0)
                for k in range(3):  # 3-class classification
                    pred[k] = 1.0 if pred[k] == val else 0.0

            return preds

        if self._patched_image:  # input composed of patches

            print("Decision fusion mechanism ...")
            images = dataframe.drop_duplicates(subset=['path'])
            print('Images found : {}'.format(len(images)))
            images_pred = np.zeros((images.shape[0], 3))

            print('Computing frequency prediction for each image patch ...')
            count = 0
            for i, image in images.iterrows():

                # print('\tImage {} : {}'.format(count, image['path']))
                patches = dataframe.loc[dataframe['path'] == image['path']]
                # print('\t\tPatch found : {}'.format(len(patches)))

                patches_pred = np.zeros((patches.shape[0], 3))

                for j, _ in patches.iterrows():
                    patch = loader(split=2, index=j)
                    patch = np.expand_dims(patch, 0)
                    patch = self._model.predict(patch)
                    patches_pred[j % patches.shape[0], :] = one_hot_encode(patch)

                prediction, frequency = np.unique(patches_pred, axis=0, return_counts=True)
                # print('\t\tPrediction :')
                # for h in range(prediction.shape[0]):
                #     print('\t\t\tClass {} : {}'.format(prediction[h], frequency[h]))

                images_pred[count, :] = prediction[np.argmax(frequency)]
                count += 1

                labels = dataframe[['label_cll', 'label_fl', 'label_mcl']].iloc[::patches.shape[0], :]

        else:  # input is directly full image

            # Extract labels of test set, predict them with the model
            images_pred = self._model.predict(test)
            images_pred = one_hot_encode(images_pred)

            labels = dataframe[['label_cll', 'label_fl', 'label_mcl']]

        # Extract labels from dataframe
        y_est_test = np.argmax(np.array(labels), axis=1)
        y_est_pred = np.argmax(np.array(images_pred), axis=1)

        cm = confusion_matrix(y_est_test, y_est_pred)
        self.plot_confusion_matrix(cm=cm, classes=['CLL', 'FL', 'MCL'], normalize=True)

        # Determine performance scores
        accuracy = accuracy_score(y_est_test, y_est_pred, normalize=True)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_est_test, y_est_pred, average='macro', zero_division=0)

        print('PERFORMANCES ON TEST SET:')
        print('Accuracy: {:.2f}%'.format(accuracy * 100))
        print('Precision: {:.2f}%'.format(precision * 100))
        print('Recall: {:.2f}%'.format(recall * 100))
        print('Fscore: {:.2f}%'.format(fscore * 100))

    # Useful plot
    def plot_curves(self):

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Curves')

        # Accuracy curve
        ax1.plot(self._history.history['categorical_accuracy'], 'b', linewidth=3.0, label='Training accuracy')
        ax1.plot(self._history.history['val_categorical_accuracy'], 'r', linewidth=3.0, label='Validation accuracy')
        ax1.set_xlabel('Iteration', fontsize=16)
        ax1.set_ylabel('Accuracy rate', fontsize=16)
        ax1.legend()
        ax1.set_title('Accuracy', fontsize=16)

        # Learning curve
        ax2.plot(self._history.history['loss'], 'b', linewidth=3.0, label='Training loss')
        ax2.plot(self._history.history['val_loss'], 'r', linewidth=3.0, label='Validation loss')
        ax2.set_xlabel('Iteration', fontsize=16)
        ax2.set_ylabel('Loss', fontsize=16)
        ax2.legend()
        ax2.set_title('Learning', fontsize=16)

        plt.show()

        # Save plot and close
        plt.savefig('plot/{}_curves.png'.format(self._name))
        plt.close()

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
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

        plt.savefig('plot/{}_confusion_matrix.png'.format(self._name))
        plt.show()

    # Property
    @property
    def is_loaded(self):
        return self._loaded



