import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

datasetdir = '/home/wojciech/Dokumenty/GSN/DogVsCatClassification'
os.chdir(datasetdir)


plt.subplot(1,2,1)
plt.imshow(img.imread('cats/cat.159.jpg'))
plt.subplot(1,2,2)
plt.imshow(img.imread('dogs/dog.231.jpg'))
plt.show()
