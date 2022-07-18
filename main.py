

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
print(query.to_string())
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


dataset_1 = dataset_1.drop(["Name","Embarked","PassengerId","Parch"],axis=1)
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


def retScore( model,xtrain,ytrain,xtest,ytest):
    model.fit(xtrain,ytrain)
    pred1 = model.predict(xtest)
    accuracy = accuracy_score(ytest,pred1)*100
    return accuracy


"""Decisio tree classifier"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model_1 = DecisionTreeClassifier()

model_1.fit(train_x,train_y)
predict_model1 = model_1.predict(test_x)

accuracy1 =accuracy_score(test_y,predict_model1)*100

print("predict_model1 :",accuracy1)


"""Random Forest classifier"""

# from sklearn.ensemble import RandomForestClassifier
#
# model_2 = RandomForestClassifier(n_estimators=1700,criterion="entropy",min_samples_split=2,max_depth= 4,max_features='sqrt',random_state=23)
#
# # model_2.fit(xtrain,ytrain)
# # predict_model2=model_2.predict(test_x)
#
# accuracy2 = retScore(model_2,train_x,train_y,test_x,test_y)#accuracy_score(test_y,predict_model2)*100
# print("Accuracy 2 :",accuracy2)
#
#
# # from sklearn.model_selection import GridSearchCV
# # rfc=RandomForestClassifier(random_state=42)
# # param_grid = {
# #     'n_estimators': [1500,1700],
# #     'max_features': [ 'sqrt', 'log2'],
# #     'max_depth' : [4,5,6,7,8],
# #     'criterion' :['gini', 'entropy']
# # }
# #
# # CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 2)
# # CV_rfc.fit(xtrain,ytrain)
# # print(CV_rfc.best_params_)
#
# """model 3  Naive baiyes algo------GaussianNB"""
#
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.naive_bayes import GaussianNB
#
# model_3 =  GaussianNB()
# # model_3.fit(xtrain,ytrain)
# # predict3 =  model_3.predict(test_x)
#
# accuracy3 = retScore(model_3,train_x,train_y,test_x,test_y)#(test_y,predict3)*100
#
# print("Accuracy 3 :",accuracy3)
#
# """GaussianProcessClassifier"""
# model_4 =  GaussianProcessClassifier(random_state=23)
# # model_4.fit(train_x,train_y)
# # predict4 =  model_4.predict(test_x)
# #
#
# accuracy4 = retScore(model_4,train_x,train_y,test_x,test_y)# accuracy_score(test_y,predict4)*100
#
# print("Accuracy 4 :",accuracy4)
#
# """KNN classifier"""
#
# from sklearn.neighbors import KNeighborsClassifier
# model_5 = KNeighborsClassifier(n_neighbors=5)
#
# accuracy5 = retScore(model_5,train_x,train_y,test_x,test_y)
#
# print("Accuracy 5 :",accuracy5)
#
#
# """Support zvector Machine"""
#
# from sklearn.svm import SVC
#
# model_6 = SVC()
# accuracy6 = retScore(model_6,train_x,train_y,test_x,test_y)
#
# print("Accuracy Score 6 :",accuracy6)
#
# """SGDClassifier"""
#
# from sklearn.linear_model import SGDClassifier
#
# model_7 = SGDClassifier()
#
# accuracy7 = retScore(model_7,train_x,train_y,test_x,test_y)
#
# all_models = pd.DataFrame({
#     "Model" : ["DecisionTreeClassifier", "RandomForestClassifier", "GaussianNB", "GaussianProcessClassifier", "KNeighborsClassifier","SVC","SGDClassifier"],
#     "Accuracy score" : [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5,accuracy6,accuracy7]
# })
#
# print(all_models.sort_values(by="Accuracy score"))
#
# """According all prediction Decision tree classifier and Random forest good score 79%"""
#
#
# """printing men and women survived percentage"""
#
# print(dataset_1.head(3).to_string())
# women = dataset_1.loc[dataset_1.Sex==0]["Embarked"]
#
# print(sum(women)/len(women))
#
# from sklearn.tree import plot_tree
# from sklearn import tree
#
# #tree.plot_tree(model_1,filled=True)
#
#
#
# print(tree.export_text(model_1))
#
# sns.barplot(dataset_2['Sex'],dataset_2['Fare'])
# plt.show()
#
# # tree.export_graphviz(model_2.estimators_[0],
# #           feature_names=ytrain,
# #           class_names=xtrain.columns,
# #           filled=True, impurity=True,
# #           rounded=True)
# #fig.savefig('figure_name.png')

# decision tree classifier is so good as comparing woth all other data models

"""Export pickle"""
# logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(train_x, train_y)
pred = model.predict(test_x)

"""third model lets check with linear regression"""

para ={
    'model':LogisticRegression(),
    'params':{
        'solver':["newton-cg","lbfgs","liblinear","sag","saga"],
        'multi_class':["auto","ovr","multinomial"],
        'penalty':["l2","elasticnet"]
    }

}
from sklearn.model_selection import  RandomizedSearchCV

# tree= RandomizedSearchCV(para,cv=5,random_state=23,scoring='accuracy')
# print(tree)
#please provide the date and time of
# model_params = {
#     'svm': {
#         'model': svm.SVR(gamma='auto'),
#         'params': {
#             'C': [1, 10, 20],
#             'kernel': ['rbf', 'linear']
#         }
#     },
# clf =None
# def retAccuracy():
#     for model_name,model_value in para.items():

        #if model_name == 'LogisticRegression':

#clf = RandomizedSearchCV(para['model'], para['params'], cv=5, return_train_score=False)

#clf.fit(train_x, train_y)


#pred1 = clf.predict(test_x)
#print("Accuracy :",accuracy_score(test_y,pred1))

#print(clf.best_params_)
"""Grid cross random searchCV"""


model3 =LogisticRegression(solver="newton-cg",penalty="l2",multi_class="auto")
model3.fit(train_x,train_y)
predicting = model3.predict(test_x)
def printgin():
    print("Vishnu try u r best")



# """K.fold validation """
# print("Accuracy :",accuracy_score(test_y,pred)*100)
# #acc
# print(xtrain.iloc[57,:])
# print(ytrain[57])
import pickle
# #
pickle.dump(open("titanic_model.pkl", 'wb'))