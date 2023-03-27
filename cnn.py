import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow import keras

datasetdir = '/home/wojciech/Dokumenty/GSN/DogVsCatClassification'
os.chdir(datasetdir)

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
gen = ImageDataGenerator()

batch_size = 30
height, width = (256,256)

imgdatagen = ImageDataGenerator(rescale = 1/255., validation_split = 0.2)
train_dataset = imgdatagen.flow_from_directory(os.getcwd(), target_size = (height, width), classes = ('dogs','cats'), batch_size = batch_size, subset = 'training')
val_dataset = imgdatagen.flow_from_directory(os.getcwd(), target_size = (height, width), classes = ('dogs','cats'), batch_size = batch_size, subset = 'validation')

model = keras.models.Sequential()
initializers = {}

model.add(keras.layers.Conv2D(24, 5, input_shape=(256,256,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(48, 5, activation='relu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(96, 5, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.9))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adamax(learning_rate=0.001), metrics=['acc'])

history = model.fit_generator(train_dataset,validation_data = val_dataset, workers=10, epochs=20)
model.save('catsvsdogs.h5')

def plot_history(history, yrange):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()


plot_history(history, (0.65, 1.))
