from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd

def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

X  = pd.read_csv("input/train.csv")
y = X.pop("Survived")

X["Age"].fillna(X.Age.mean(), inplace = True)
X["Cabin"] = X.Cabin.apply(clean_cabin)

categorical_variables = ["Sex", "Cabin", "Embarked"]

for variable in categorical_variables:
    X[variable].fillna("Missing", inplace = True)
    dummies = pd.get_dummies(X[variable], prefix=variable)
    X = pd.concat([X, dummies], axis = 1)
    X.drop([variable], axis=1, inplace = True)


X.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)
# print X
# numeric_variables = list(X.dtypes[X.dtypes != "object"].index)

model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)

y_oob = model.oob_prediction_
print "c_stat : ", roc_auc_score(y, y_oob)

print model.feature_importances_`