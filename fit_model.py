from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNLSTM
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions

import numpy as np
import pandas as pd

################### loading data ###################
x_data = np.load('/home/edward/Documents/ML/hotdogs/data/x_data.npy')

y_data = np.load('/home/edward/Documents/ML/hotdogs/data/y_data.npy')

################### setting up pre-trained model ###################
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense

model = InceptionV3(weights='imagenet')

intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[311].output)

x = intermediate_layer_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

transfer_model = Model(inputs=intermediate_layer_model.input, outputs=predictions)

# train last cluster and dense layer
for layer in transfer_model.layers:
    layer.trainable = False

for i in range(280,313):
	transfer_model.layers[i].trainable = True

# [4.555431608200073, 0.39960000000000001, 0.63949999999999996]

transfer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])


################### training last two layers ###################
epoch = 20
num_classes = 2
# onehot encoding
y_onehot = np.zeros((y_train.shape[0], num_classes))
for i in range(0,num_classes):
	(y_onehot[:,i:i+1])[y_train==i] = 1

# using checkpoints and early stopping on validation sample to prevent overfitting
# best weight is saved to file_path
file_path="/home/edward/Documents/ML/hotdogs/weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
callbacks_list = [checkpoint, early] #early


transfer_model.fit(x_train, y_onehot, epochs=epoch, validation_split=0.1, batch_size=32, callbacks=callbacks_list)
# load best weights
transfer_model.load_weights(file_path)

################### predicting given image ###################
to_pred = '/home/edward/Documents/ML/hotdogs/data/raw/tests/test_image.jpg'
img = image.load_img(path, target_size=(250, 250))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = transfer_model.predict(x)
print("The image is this likely to be a hotdog", preds[0][1])