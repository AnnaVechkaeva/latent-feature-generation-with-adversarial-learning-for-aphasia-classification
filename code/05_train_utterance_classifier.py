# Step 5 - Utterance-level classifier (paper Sec. 3.4.1)
# Flatten (28x300) -> Dense(128, ReLU) -> Dense(4, softmax), trained for 5 epochs.
# Train on real data (baseline) or on real + generated data by changing the input files
# (see commented-out lines). Saves the model as .h5.
# Note: test_images / test_labels are commented out, so the validation_data line needs
# them to be loaded first (as was done in 03_gan_experiments.ipynb).
#
# Original file name: CNN_classification.py (2020). Code below is unchanged.

from __future__ import absolute_import, division, print_function
import json

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

train_images = np.load("aphasia_np_vectors_patientwise_train.np.npy")
#train_images = nliew_train_vectors
#train_images = np.load("wgan_2000_vectors_half.npy")
train_labels = np.load("aphasia_np_labels_num_patientwise_train.np.npy")
#train_labels = new_train_labels
#train_labels = np.load("wgan_2000_labels_half.npy")

#test_images = np.load("aphasia_np_vectors_patientwise_test.np.npy")[1:]
#test_labels = np.load("aphasia_np_labels_num_patientwise_test.np.npy")

print("Data loaded")

#print(test_labels.shape, test_images.shape)
print(train_labels.shape, train_images.shape)

seed_value = 1
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
#import tensorflow as tf
#tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.get_session()



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 300)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
#history = model.fit(train_images, train_labels, shuffle=False, epochs=10, validation_data=(test_images[1:], test_labels))
#model.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


#test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


from keras.models import model_from_json

# Save the weights
model.save('CNN_bal2000_half.h5')
print("Saved model to disk")
