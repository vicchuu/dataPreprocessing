

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dataset_1 = pd.read_csv("titanic_train.csv")

#dataset_1["Age"].fillna(90)
#
#print(dataset_1.info())
#
# print(dataset_1.isnull().sum())
#
# print(dataset_1.describe())
#
# print(dataset_1["Name"].values())

for col in dataset_1:
    print("*******Start********",col )
    print(dataset_1[col].value_counts())
    print("*********END ******",col)
    print()
    print()

plt.figure(figsize=(24,8))
sns.heatmap(dataset_1.isnull(),cmap='YlGnBu')

normal_data = np.random.randint(1,10,(10, 12))
ax = sns.heatmap(normal_data, center=0,cmap="YlGnBu")
#plt.show()

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

print(query.shape)
for a in query["Age"]:
    print(a)
#print(query[:10])
#print(data_query[:10])

#plt.show()

print(dataset_1.sort_values('Age',ascending=True).head(10).to_string())

#for a in dataset_1:
det = dataset_1.loc["Age"]
for a in det:

    print(a)



