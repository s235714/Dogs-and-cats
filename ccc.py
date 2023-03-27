import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow import keras

datasetdir = '/home/wojciech/Dokumenty/GSN/DogVsCatClassification'
os.chdir(datasetdir)

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
gen = ImageDataGenerator()



iterator = gen.flow_from_directory(
    os.getcwd(),
    target_size=(256,256),
    classes=('dogs','cats')
)

batch = iterator.next()
len(batch)

plt.imshow(batch[0][0].astype(np.int))
plt.show()
