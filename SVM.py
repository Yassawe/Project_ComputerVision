import pickle
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('pixel_values.csv', header=None)

data=data.sample(frac=1)
X = data.iloc[:, 0:4]
Y = data.iloc[:, 4:5]

model = SVC(gamma='auto')
model.fit(X, Y)

pickle.dump(model, open('model.pickle', 'wb'))