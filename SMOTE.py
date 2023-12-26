import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter

data = pd.read_csv("car_evaluation.csv")
# print(data.head())
# print(data.shape)


# print(data.outcome.value_counts())

x = data.iloc[:,:-1]
y = data.outcome
# print(x.head())

enc = LabelEncoder()
x.loc[:,['buying','maint','lug_boot','safety']] = x.loc[:,['buying','maint','lug_boot','safety']].apply(enc.fit_transform)
# print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)

model = KNeighborsClassifier()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

print(accuracy_score(y_test,y_predict))
print(pd.crosstab(y_test,y_predict))

smote = SMOTE()

x_train_smote,y_train_smote = smote.fit_resample(x_train.astype('float'),y_train)

print("Before SMOTE :",Counter(y_train))
print("\nAfter  SMOTE :",Counter(y_train_smote))

model.fit(x_train_smote,y_train_smote)
y_predict = model.predict(x_test)
print(accuracy_score(y_test,y_predict))
print(pd.crosstab(y_test,y_predict))