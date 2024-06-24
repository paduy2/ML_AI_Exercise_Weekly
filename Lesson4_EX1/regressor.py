import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


target = "math score"
data = pd.read_csv("StudentScore.xls")

#Split data

x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Compose pipeline for transform each type of catego data

# Num type
num_transform = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler()),
])

# Ord type
education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
                    "master's degree"]
gender = ["male", "female"]
lunch = x_train["lunch"].unique()
test_prep = x_train["test preparation course"].unique()
ord_transform = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('encoder', OrdinalEncoder(categories=[education_values, gender, lunch, test_prep])),
])
# Nom type
nom_transform = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('encoder', OneHotEncoder(sparse_output=False,handle_unknown="ignore")),
])

#Compose prepocessor
preprocessor = ColumnTransformer(transformers = [
  ('num_feature', num_transform, ["reading score", "writing score"]),
  ('ord_feature', ord_transform, ["parental level of education", "gender", 
                                  "lunch", "test preparation course"]),
  ('nom_feature', nom_transform, ["race/ethnicity"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
for i, j in zip(y_test, y_predict):
  print("Actual: {} . Predict:{} ".format(i,j))