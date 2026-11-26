# Step 4 - Generating synthetic utterance vectors with a trained generator (paper Sec. 3.5.2)
# Several experiment variants are kept as commented-out blocks:
#   * 10,000 generated datapoints per class (training only on generated data, Table 5)
#   * generating for the minority classes until every class has as many datapoints as
#     Control (real + generated, Table 4 and later experiments)
# Change the generator file / input files / output names to switch between experiments.
#
# Label ids: 0 = Anomic, 1 = Broca, 2 = Conduction, 3 = Control.
#
# Original 2019 code below, unchanged.

import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import sys


train_images = np.load("train1_vectors.npy")
train_labels = np.load("train1_labels.npy")

GANmodel = tf.keras.models.load_model("cGAN_generator1_pw_20000.h5")

'''
labels = np.array([0]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs0 = GANmodel.predict([noise, labels])

labels = np.array([1]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs1 = GANmodel.predict([noise, labels])

labels = np.array([2]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs2 = GANmodel.predict([noise, labels])

labels = np.array([3]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs3 = GANmodel.predict([noise, labels])


new_train_vectors = np.append(train_images, gen_imgs0.reshape(10000, 28, 300), axis=0)
print(0)
new_train_vectors = np.append(new_train_vectors, gen_imgs1.reshape(10000, 28, 300), axis=0)
print(1)
new_train_vectors = np.append(new_train_vectors, gen_imgs2.reshape(10000, 28, 300), axis=0)
print(2)
new_train_vectors = np.append(new_train_vectors, gen_imgs3.reshape(10000, 28, 300), axis=0)
print(3)



new_train_labels = np.append(train_labels,np.array([1]*10000), axis=0)
new_train_labels = np.append(new_train_labels,np.array([1]*10000), axis=0)
new_train_labels = np.append(new_train_labels,np.array([2]*10000), axis=0)
new_train_labels = np.append(new_train_labels,np.array([3]*10000), axis=0)

np.save("aphasia_np_labels_patientwise_train_bal_half.np", new_train_labels)
np.save("aphasia_np_vectors_patientwise_train_bal_half.np", new_train_vectors)
'''
'''
n = 10000

labels = np.array([0]*n)
noise = np.random.normal(0, 1, (n, 100))
gen_imgs0 = GANmodel.predict([noise, labels])

labels = np.array([1]*n)
noise = np.random.normal(0, 1, (n, 100))
gen_imgs1 = GANmodel.predict([noise, labels])

labels = np.array([2]*n)
noise = np.random.normal(0, 1, (n, 100))
gen_imgs2 = GANmodel.predict([noise, labels])

labels = np.array([3]*n)
noise = np.random.normal(0, 1, (n, 100))
gen_imgs3 = GANmodel.predict([noise, labels])


new_train_vectors = gen_imgs0.reshape(n, 28, 300)
print(0)
new_train_vectors = np.append(new_train_vectors, gen_imgs1.reshape(n, 28, 300), axis=0)
print(1)
new_train_vectors = np.append(new_train_vectors, gen_imgs2.reshape(n, 28, 300), axis=0)
print(2)
new_train_vectors = np.append(new_train_vectors, gen_imgs3.reshape(n, 28, 300), axis=0)
print(3)

new_train_labels = np.array([0]*n)
new_train_labels = np.append(new_train_labels,np.array([1]*n), axis=0)
new_train_labels = np.append(new_train_labels,np.array([2]*n), axis=0)
new_train_labels = np.append(new_train_labels,np.array([3]*n), axis=0)
'''
'''
#np.save("aphasia_np_labels_patientwise_train_20k.np", new_train_labels)
#np.save("aphasia_np_vectors_patientwise_train_20k.np", new_train_vectors)
'''

dp_num = dict()
for l in train_labels:
    if l in dp_num:
        dp_num[l] += 1
    else:
        dp_num[l] = 1

nums = [dp_num[0], dp_num[1], dp_num[2], dp_num[3]]

max_num = max(nums)

num0 = max_num - dp_num[0]
num1 = max_num - dp_num[1]
num2 = max_num - dp_num[2]
num3 = max_num - dp_num[3]

labels = np.array([0]*num0)
noise = np.random.normal(0, 1, (num0, 100))
gen_imgs0 = GANmodel.predict([noise, labels])

labels = np.array([1]*num1)
noise = np.random.normal(0, 1, (num1, 100))
gen_imgs1 = GANmodel.predict([noise, labels])

labels = np.array([2]*num2)
noise = np.random.normal(0, 1, (num2, 100))
gen_imgs2 = GANmodel.predict([noise, labels])

#labels = np.array([3]*num3)
#noise = np.random.normal(0, 1, (num3, 100))
#gen_imgs3 = GANmodel.predict([noise, labels])

new_train_vectors = np.append(train_images, gen_imgs0.reshape(num0, 28, 300), axis=0)
print(0)
new_train_vectors = np.append(new_train_vectors, gen_imgs1.reshape(num1, 28, 300), axis=0)
print(1)
new_train_vectors = np.append(new_train_vectors, gen_imgs2.reshape(num2, 28, 300), axis=0)
print(2)
#new_train_vectors = np.append(new_train_vectors, gen_imgs3.reshape(num3, 28, 300), axis=0)
#print(3)

new_train_labels = np.append(train_labels,np.array([0]*num0), axis=0)
new_train_labels = np.append(new_train_labels,np.array([1]*num1), axis=0)
new_train_labels = np.append(new_train_labels,np.array([2]*num2), axis=0)
#new_train_labels = np.append(new_train_labels,np.array([3]*num3), axis=0)
'''
labels = np.array([0]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs0 = GANmodel.predict([noise, labels])

labels = np.array([1]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs1 = GANmodel.predict([noise, labels])

labels = np.array([2]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs2 = GANmodel.predict([noise, labels])

labels = np.array([3]*10000)
noise = np.random.normal(0, 1, (10000, 100))
gen_imgs3 = GANmodel.predict([noise, labels])

new_train_vectors = np.append(new_train_vectors, gen_imgs0.reshape(10000, 28, 300), axis=0)
print(0)
new_train_vectors = np.append(new_train_vectors, gen_imgs1.reshape(10000, 28, 300), axis=0)
print(1)
new_train_vectors = np.append(new_train_vectors, gen_imgs2.reshape(10000, 28, 300), axis=0)
print(2)
new_train_vectors = np.append(new_train_vectors, gen_imgs3.reshape(10000, 28, 300), axis=0)
print(3)

new_train_labels = np.append(new_train_labels,np.array([1]*10000), axis=0)
new_train_labels = np.append(new_train_labels,np.array([1]*10000), axis=0)
new_train_labels = np.append(new_train_labels,np.array([2]*10000), axis=0)
new_train_labels = np.append(new_train_labels,np.array([3]*10000), axis=0)
'''
np.save("wgan_2000_labels_half", new_train_labels)
np.save("wgan_2000_vectors_half", new_train_vectors)

