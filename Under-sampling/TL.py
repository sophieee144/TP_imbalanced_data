from imblearn.under_sampling import TomekLinks
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/hcv_classification.csv")

X = df_train.drop(['Activity'],axis=1)
y = df_train['Activity']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# Visualize the original data distribution
sns.countplot(x = y_train)
plt.title("Original Class Distribution")
plt.show()

# Apply Tomek Links undersampling
tl = TomekLinks()

X_resampled, y_resampled = tl.fit_resample(X_train, y_train)

print("Before TOMEK LINKS :",Counter(y_train))
print("\nAfter  TOMEK LINKS :",Counter(y_resampled))

# Visualize the resampled data distribution
sns.countplot(x = y_resampled)
plt.title("Class Distribution after Tomek Links Undersampling")
plt.show()