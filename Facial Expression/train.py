import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

x_train = np.load('./data/feature_vectors/features.npy')
with open('./data/feature_vectors/labels.txt', 'r') as f:
    y_train = np.array([line.strip() for line in f.readlines()])
    
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', probability=True)
)

svm_model.fit(x_train, y_train)

joblib.dump(svm_model, './data/svm_model.joblib')

print(accuracy_score(y_train, svm_model.predict(x_train)))