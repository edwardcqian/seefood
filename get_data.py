import numpy as np
import pandas as pd

import urllib.request
import re

import os
from os import listdir
from os.path import isfile, join

from PIL import Image

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input


################### sampling "not hot dog" images from ImageNet ###################
# src_file contains a list of urls for the images (given by ImageNet) 
# saves to destination as 1.jpg, 2.jpg, ...
src_file = ".../data/raw/images.txt"
dest = '.../data/raw/random'

with open(src_file) as file:
	i = 0
	for line in file:
		if i == 5000:
			break
		line = line.strip()
		line = re.findall(r'(https?://[^\s]+)', line)[0]
		print(i, line)
		try:
			with Timeout(10):
				urllib.request.urlretrieve(line, dest+"/%d.jpg" % (i))
		except:
 			print("Problem on line ", line)
		i += 1

# check if images can be opened, delete corrupt files
count = 0
onlyfiles = [f for f in listdir(dest) if isfile(join(dest, f))]
for f in onlyfiles:
    try:
        Image.open(dest+'/'+f)
    except:
        count += 1
        try:
            os.remove(dest+'/'+f)
        except OSError:
            pass
print(count, " images removed")

################### loading images as RGB PIL format ###################
not_path = '.../data/raw/not_hotdog'
not_hotdog = [f for f in listdir(not_path) if isfile(join(not_path, f))]

is_path = '.../data/raw/is_hotdog'
is_hotdog = [f for f in listdir(is_path) if isfile(join(is_path, f))]

# size of image
size = 250, 250

# initalize x
x_data = np.zeros((len(not_hotdog)+len(is_hotdog), 250, 250, 3))

# load not hotdogs
for i, f in enumerate(not_hotdog):
    path = not_path+'/'+f
    try:
        img = image.load_img(path, target_size=(size))
    except:
        print(path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x_data[i] = x

# load hotdogs
for i, f in enumerate(is_hotdog):
    path = is_path+'/'+f
    img = image.load_img(path, target_size=(size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x_data[i+len(not_hotdog)] = x

# create labels 
y_data = np.zeros(len(not_hotdog)+len(is_hotdog))
y_data[len(not_hotdog):] = 1


# saving data
np.save('.../x_data.npy', x_data)
np.save('.../y_data.npy', y_data)
