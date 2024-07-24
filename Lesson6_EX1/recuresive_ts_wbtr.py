import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function for creating data which is appreciate with window_size 

def create_recursive_data(data, window_size):
  i = 1
  while i < window_size:
    data["users_{}".format(i)] = data["users"].shift(-i)
    i += 1
  data["target"] = data["users"].shift(-i)
  data = data.dropna(axis=0)
  return data

data = pd.read_csv("web-traffic.csv")
data["date"] = pd.to_datetime(data["date"] , format='%d/%m/%y')
data["users"] = data["users"].interpolate()
#Visual dataset
# fig, ax = plt.subplots()
# ax.plot(data["date"], data["users"])
# ax.set_xlabel("Time")
# ax.set_ylabel("Users")
# plt.show()

window_size = 5
data = create_recursive_data(data, window_size)
x = data.drop(["date", "target"], axis=1)
y = data["target"]
train_size = 0.8
num_samples = len(data)
x_train = x[:int(train_size*num_samples)]
y_train = y[:int(train_size*num_samples)]
x_test = x[int(train_size*num_samples):]
y_test = y[int(train_size*num_samples):]


model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

#Visual the result

# fig, ax = plt.subplots()
# ax.plot(data["time"][:int(train_size*num_samples)], data["co2"][:int(train_size*num_samples)], label="train")
# ax.plot(data["time"][int(train_size*num_samples):], data["co2"][int(train_size*num_samples):], label="test")
# ax.plot(data["time"][int(train_size*num_samples):], y_predict, label="prediction")
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# ax.legend()
# ax.grid()
# plt.show()

#Make the predict
exit ()
current_data = [2520, 3150, 5421, 4300, 3422]
for i in range(10):
    print(current_data)
    prediction = model.predict([current_data]).tolist()
    print("Traffice in week {} is {}".format(i+1, prediction[0]))
    print("-----------------")
    current_data = current_data[1:] + prediction