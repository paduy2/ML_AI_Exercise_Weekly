import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import joblib

target = "result"
non_scale = "map"
data = pd.read_csv("csgo.csv")
# GET data summarization
# x = data.drop([non_scale,"day","month","year","date","wait_time_s","match_time_s","team_a_rounds","team_b_rounds"], axis=1)
# y = x.replace({'Win': 1, 'Lost': -1, 'Tie': 0})
# print (y)
# profile = ProfileReport(y, title="Match Report", explorative=True)
# profile.to_file("report.html")

# split data

x = data.drop([target,"day","month","year","date","wait_time_s","match_time_s","team_a_rounds","team_b_rounds"], axis=1)
y = data[target]
# y = data[target].replace({'Win': 1, 'Lose': -1, 'Tie': 0})


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Separate the DataFrame into the columns to be scaled and the column to exclude
x_train_non_scale = x_train[non_scale]
x_train_to_scale = x_train.drop(columns=[non_scale], axis=1)

x_test_non_scale = x_test[non_scale]
x_test_to_scale = x_test.drop(columns=[non_scale], axis=1)
# Preprocess data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_to_scale)
x_test = scaler.transform(x_test_to_scale)

#Prepare params for randomforest model
params = {
    "n_estimators": [50, 100],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 5],
    "min_samples_split": [2, 5, 10]
}

# model = SVC()
model = LogisticRegression()
# model = RandomForestClassifier()
model.fit(x_train, y_train)

# model = GridSearchCV(RandomForestClassifier(random_state=100), param_grid=params, scoring="precision", cv=6, verbose=2,n_jobs=1)
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)

#Analyze 
y_predict = model.predict(x_test)

# Get predicted probabilities
y_prob = model.predict_proba(x_test)

# Get the class labels
class_labels = model.classes_
print("Class labels:", class_labels)
# Set custom thresholds for each class
thresholds = [0.2, 0.2, 0.6]  # Adjust these thresholds based on your needs

# Function to apply thresholds and make predictions
def apply_thresholds(y_prob, thresholds):
    y_pred = np.zeros_like(y_prob)
    for i, threshold in enumerate(thresholds):
        y_pred[:, i] = (y_prob[:, i] >= threshold).astype(int)
    return y_pred.argmax(axis=1)

# Apply thresholds to the predicted probabilities
y_pred_custom_thresholds = apply_thresholds(y_prob, thresholds)
# print (y_pred_custom_thresholds)
# print (type(y_pred_custom_thresholds))

# Map numeric predictions back to original labels
y_pred_mapped = class_labels[y_pred_custom_thresholds]
#Show Result
# for i,j in zip(y_predict, y_test):
#   print ("Predict: {} . Actual: {}".format(i,j))
print(classification_report(y_test, y_predict))
print(classification_report(y_test, y_pred_mapped, target_names=class_labels))
# # Visualization
# print (confusion_matrix(y_test,y_predict))
cm = np.array(confusion_matrix(y_test,y_predict))
confusion = pd.DataFrame(cm, index=["Win", "Tie", "Lost"], columns=["Win", "Tie", "Lost"])
sns.heatmap(confusion, annot=True)
plt.show()

# Save the model
# joblib_file = "logistic_regression_model.pkl"
# joblib.dump(model, joblib_file)

# Load the model
# loaded_model = joblib.load(joblib_file)