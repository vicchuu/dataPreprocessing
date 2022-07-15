

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dataset_1 = pd.read_csv("titanic_train.csv")


# dataset_1 =  dataset_1.replace({
#     np.Nan :0
# })
#print(dataset_1.head(10).to_string())
#new_data = dataset_1["Age"].fillna(0)
#
#print(dataset_1.info())
#
# print(dataset_1.isnull().sum())
#
# print(dataset_1.describe())
#
# print(dataset_1["Name"].values())

# for col in dataset_1:
#     print("*******Start********",col )
#     print(dataset_1[col].value_counts())
#     print("*********END ******",col)
#     print()
#     print()

plt.figure(figsize=(24,8))
sns.heatmap(dataset_1.isnull(),cmap='YlGnBu')

# normal_data = np.random.randint(1,10,(10, 12))
# ax = sns.heatmap(normal_data, center=0,cmap="YlGnBu")
# #plt.show()

# temp=0
# for v in dataset_1["Survived"]:
#     if v==0 & dataset_1["Age"]==22:
#         temp+=1
#         print(temp)

# for a in dataset_1["Age"]:
#     if((a.Survived==1) & (a.Age <=30)):
#         print(a)

query = pd.DataFrame(dataset_1[(dataset_1["Age"]<30) & dataset_1["Survived"]==1])

#query1 = dataset_1.groupby()

#print(query.shape)
for a in query["Age"]:
    print(a)
#print(query[:10])
#print(data_query[:10])

#plt.show()

#print(dataset_1.sort_values('Age',ascending=True).head(10).to_string())

# #for a in dataset_1:
# det = dataset_1.loc["Age"]
# for a in det:
#
#     print(a)

#print(dataset_1.head(10).to_string())

"""lets fill up empty values in training set"""

#print(dataset_1.isnull().sum().sort_values(ascending=False))

#print(dataset_1.info())
"""Cabin == 687(object) values and Age == 177(float64), embarked ==2(object)"""

def fillMissingValues(dataset_1):

    for a in dataset_1.columns:

        if dataset_1[a].dtypes =='object':
            dataset_1[a].fillna(dataset_1[a].mode()[0],inplace =True)
        else:
            dataset_1[a].fillna(dataset_1[a].median(),inplace=True)
    return dataset_1

dataset_2 = fillMissingValues(dataset_1)

#print(dataset_2.isnull().sum().sort_values(ascending=True))
"""Vishnu good u has done..."""

print(dataset_2.describe().to_string())


"""Auto is not working in pycharm """
# plt.figure(figsize = (10, 5))
# from autoviz.AutoViz_Class import AutoViz_Class
# AV = AutoViz_Class()
# #df_av = AV.AutoViz("titanic_train.csv")
#
# sep = ","
# dft = AV.AutoViz(
# "titanic_train.csv",
# sep=",",
# depVar="",
# dfte=None,
# header=0,
# verbose=0,
# lowess=False,
# chart_format="svg",
# max_rows_analyzed=150000,
# max_cols_analyzed=30,
# )
# plt.show()

"""Label Encoding"""

from sklearn.preprocessing import LabelEncoder

label =LabelEncoder()

for a in dataset_2.columns:
    if dataset_2[a].dtypes =='object':
        dataset_2[a] = label.fit_transform(dataset_2[a])

print(dataset_2.describe().to_string())

print(dataset_2.corr().to_string())

sns.heatmap(dataset_2.corr(),cmap="twilight",annot=True)#YlGnBu
#plt.show()


"""Preparing trian model and test model"""

xtrain = dataset_2.drop(columns="Survived",axis=1)
ytrain = dataset_2["Survived"]

from sklearn.model_selection import train_test_split
train_x , test_x, train_y,test_y = train_test_split(xtrain,ytrain,test_size=0.2,random_state=1)

print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)

"""Decisio tree classifier"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model_1 = DecisionTreeClassifier()

model_1.fit(train_x,train_y)
predict_model1 = model_1.predict(test_x)

accuracy1 = accuracy_score(test_y,predict_model1)*100

print("predict_model1 :",accuracy1)


"""Random Forest classifier"""

from sklearn.ensemble import RandomForestClassifier

model_2 = RandomForestClassifier(n_estimators=1700,criterion="entropy",min_samples_split=2,max_depth= 4,max_features='sqrt',random_state=23)

model_2.fit(xtrain,ytrain)
predict_model2=model_2.predict(test_x)

accuracy2 = accuracy_score(test_y,predict_model2)*100
print("Accuracy 2 :",accuracy2)


# from sklearn.model_selection import GridSearchCV
# rfc=RandomForestClassifier(random_state=42)
# param_grid = {
#     'n_estimators': [1500,1700],
#     'max_features': [ 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }
#
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 2)
# CV_rfc.fit(xtrain,ytrain)
# print(CV_rfc.best_params_)

"""model 3  Naive baiyes algo------GaussianNB"""

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

model_3 =  GaussianNB()
model_3.fit(xtrain,ytrain)
predict3 =  model_3.predict(test_x)

accuracy3 = accuracy_score(test_y,predict3)*100

print("Accuracy 3 :",accuracy3)

"""GaussianProcessClassifier"""
model_4 =  GaussianProcessClassifier(random_state=23)
model_4.fit(xtrain,ytrain)
predict4 =  model_4.predict(test_x)

accuracy4 = accuracy_score(test_y,predict4)*100

print("Accuracy 4 :",accuracy4)

