import pickle
from sklearn.preprocessing import StandardScaler
def findPrediction(arr):

    my_model = pickle.load(open("titanic_model.pkl",'rb'))
    x=arr
    #print(my_model.get_booster().feature_names)
    prediction = my_model.predict(x)
    #my_model.printgin()
    if prediction ==0:
        print("Sorry thaat person is dead ..")
    else:
        print("that day he is not dead, may be as soon")
    print("Prediction :",prediction)

x=[3.0000,1.000,2.500,0.000,215.000,7.2292,47.000]

y = StandardScaler().fit_transform([x])

findPrediction(y)