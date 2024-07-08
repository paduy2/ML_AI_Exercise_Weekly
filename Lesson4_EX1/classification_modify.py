import numpy as np
import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

# Preprocess data

# Compose pipeline for transform each type of catego data

# Num type
num_transform = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler()),
])

# Ord type

# Define the list for ordinal colummns
#<for replacing>
# education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
#                     "master's degree"]
# gender = ["male", "female"]
# lunch = x_train["lunch"].unique()
# test_prep = x_train["test preparation course"].unique()

# ord_transform = Pipeline(steps=[
#   ('imputer', SimpleImputer(strategy='most_frequent')),
#   ('encoder', OrdinalEncoder(categories=[education_values, gender, lunch, test_prep])),
# ])

# Nom type
nom_transform = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('encoder', OneHotEncoder(sparse_output=False,handle_unknown="ignore")),
])


#Compose prepocessor
preprocessor = ColumnTransformer(transformers = [
  ('num_feature', num_transform, ["ping","kills","assists","deaths","mvps","hs_percent","points"]),
  # ('ord_feature', ord_transform, [""]),
  ('nom_feature', nom_transform, ["map"]),
])



#Prepare params for randomforest model
# params = {
#     "n_estimators": [50, 100],
#     "criterion": ["gini", "entropy", "log_loss"],
#     "max_depth": [None, 2, 5],
#     "min_samples_split": [2, 5, 10]
# }

# model = SVC()
# model = LogisticRegression()
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

#Initialize model for model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

#Visual RESULT

# for i, j in zip(y_test, y_predict):
#   print("Actual: {} . Predict:{} ".format(i,j))

# print (confusion_matrix(y_test,y_predict))
# cm = np.array(confusion_matrix(y_test,y_predict))
# confusion = pd.DataFrame(cm, index=["Win", "Tie", "Lost"], columns=["Win", "Tie", "Lost"])
# sns.heatmap(confusion, annot=True)
# plt.show()

"""Print report confusion matrix"""
report = classification_report(y_test, y_predict, target_names=['Win', 'Tie','Lost'])
print(report)