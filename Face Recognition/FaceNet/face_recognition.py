from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from sklearn.metrics import accuracy_score

data = load('./preprocessed_data/test_data.npz')
testX_faces = data['arr_0']
    
train_data = load('./preprocessed_data/train_embeddings.npz')
trainX, trainY = train_data['arr_0'], train_data['arr_1']
test_data = load('./preprocessed_data/test_embeddings.npz')
testX, testY = test_data['arr_0'], test_data['arr_1']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

all_labels = list(trainY) + list(testY)
out_encoder = LabelEncoder()
out_encoder.fit(all_labels)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = accuracy_score(trainY, yhat_train)
score_test = accuracy_score(testY, yhat_test)
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testY[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()