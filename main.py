import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("Heart_Disease_Prediction.csv")

Xf=["Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120","EKG results","Max HR","Exercise angina","ST depression","Slope of ST","Number of vessels fluro","Thallium"]
yf=["Heart Disease"]

X_Train,X_Test,y_train,y_test=train_test_split(data[Xf],data[yf],test_size=0.2,random_state=42)
y_train=y_train.values.ravel()

model=LogisticRegression(max_iter=1000)
model.fit(X_Train,y_train)

y_pred=model.predict(X_Test)

accuracy=accuracy_score(y_test,y_pred)

Xz=[]
for i in Xf:
    x=int(input("Enter "+i+" : "))
    Xz.append(x)
    
Xz=np.array(Xz).reshape(1,-1)
Yz=model.predict(Xz)

print(Yz)

print("Accuracy : ",accuracy*100)

