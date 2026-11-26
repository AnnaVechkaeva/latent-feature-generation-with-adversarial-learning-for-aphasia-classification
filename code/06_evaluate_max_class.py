# Step 6 - Subject-level evaluation with the Max Class model (paper Sec. 3.4.2, 3.5.3, Tables 4-7)
# Classifies every test utterance with the trained utterance classifier, then assigns each
# subject the most frequent predicted class. The number of Control (label 3) predictions is
# divided by a constant: change `d[3] = d[3]/1` to /2 ... /7 for MC-2 ... MC-7.
# Prints accuracy, confusion matrix, classification report (F1) and per-class accuracy.
#
# Original 2020 code below, unchanged.

import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import operator

CNN_class = tf.keras.models.load_model("CNN_pw_half.h5")

vectors_file = "aphasia_np_vectors_patientwise_test.np.npy"
labels_file = "aphasia_np_labels_num_patientwise_test.np.npy"
subjId_file = "aphasia_np_subj_id_patientwise_test.np.npy"

vectors = np.load(vectors_file)
labels = np.load(labels_file)
subjId = np.load(subjId_file)

subjDict = dict()
subjIdDict = dict()
subjLabel = dict()

for n in range(len(labels)):
    vec = np.array([vectors[n]])
    lab = labels[n]
    sub = subjId[n]
    prediction = CNN_class.predict(vec)
    pred_lab = prediction.argmax(axis=-1)[0]
    if sub in subjDict:
        subjDict[sub].append(pred_lab)
    else:
        subjDict[sub] = [pred_lab]
        subjLabel[sub] = lab

correct = 0
incorrect = 0

y_true = []
y_pred = []


for subj in subjDict:
    d = dict()
    preds = subjDict[subj]
    for p in preds:
        if p in d:
            d[p] += 1
        else:
            d[p] = 1
    if 3 in d:
        d[3] = d[3]/1
    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    prediction = sorted_d[::-1][0][0]
    gt = subjLabel[subj]
    y_true.append(gt)
    y_pred.append(prediction)
    if prediction == gt:
        correct += 1
    else:
        incorrect += 1
#for i in range(len(y_true)):
#    print(y_true[i], y_pred[i])

print(correct, incorrect, correct/(correct+incorrect))

from sklearn import metrics

print(metrics.confusion_matrix(y_true, y_pred))

print(metrics.classification_report(y_true, y_pred, digits=3))

acc_dict = dict()
for i in range(len(y_pred)):
    if y_true[i] in acc_dict:
        acc_dict[y_true[i]].append(y_pred[i])
    else:
        acc_dict[y_true[i]] = [y_pred[i]]


print("Per class accuracy:")
for i in acc_dict:
    cor = 0
    incor = 0
    preds = acc_dict[i]
    for p in preds:
        if i == p:
            cor += 1
        else:
            incor += 1
    print(i, cor/(cor+incor))

