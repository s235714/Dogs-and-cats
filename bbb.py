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

imgdatagen = ImageDataGenerator(rescale = 1/255., horizontal_flip = True, zoom_range = 0.3, rotation_range = 15., validation_split = 0.1)



image = img.imread('cats/cat.14.jpg')

def plot_transform():
    '''apply the transformation 8 times randomly'''
    nrows, ncols = 2,4
    fig = plt.figure(figsize=(ncols*3, nrows*3), dpi=90)
    for i in range(nrows*ncols):
        timage = imgdatagen.random_transform(image)
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(timage)
        plt.axis('off')
        plt.show()

plot_transform()
