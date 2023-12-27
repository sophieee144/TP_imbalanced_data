import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/hcv_classification.csv")
print(df.head())

x = df.drop(['Activity'],axis=1)
y = df.Activity

y.value_counts().plot.pie(autopct='%2f')

plt.show()

ros = RandomOverSampler(sampling_strategy= 1)

x_res,y_res = ros.fit_resample(x,y)

ax = y_res.value_counts().plot.pie(autopct='%2f')
_ = ax.set_title('over-sampling')
plt.show()