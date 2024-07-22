import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

with np.load("mnist.npz", "r") as data:
  # print (data["x_train"].shape, data["x_test"].shape)
  # print (data["y_train"].shape, data["y_test"].shape)
  x_train = data["x_train"]
  x_test = data["x_test"]
  y_train = data["y_train"]
  y_test = data["y_test"]

#Flaten the image 
  x_train = np.reshape(x_train, (x_train.shape[0], -1))
  x_test = np.reshape(x_test, (x_test.shape[0], -1))

  print (x_train.shape)
  print (x_test.shape)
  print (y_train.shape)
  print (y_test.shape)

  cls = DecisionTreeClassifier()

  cls.fit(x_train, y_train)

  y_predict = cls.predict(x_test)
  print (classification_report(y_test, y_predict))