from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
import sys
import os

import keras.backend as K
K.set_image_data_format('channels_last')

from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from matplotlib.pyplot import imshow


# Check i/nput arguments
# if len(sys.argv) != 2:
#     print("Error: Image path missing")
#     exit()
# if not os.path.isdir(sys.argv[1]):
#     print("Error: Image does not exist")
#     exit()
imagePath = sys.argv[1]


model = load_model('my_model.h5')

# testGen = ImageDataGenerator(rescale=1. / 255)
# testImage = testGen.flow_from_directory(
#     imagePath,
#     class_mode='categorical')

# prediction = model.predict_generator(testImage)
# print(prediction)


### END CODE HERE ###
img = image.load_img(imagePath, target_size=(256, 256))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

output = model.predict(x)

out = ""
if output[0][0] == 1:
    out = "busy"
elif output[0][1] == 1:
    out = "clean"
elif output[0][2] == 1:
    out = "messy"
else :
    out = "ERROR"
print("The lab is:  " + out)