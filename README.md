# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
   

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: N Preethika
RegisterNumber:  212223040130
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

Head

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/260ff8e9-59ef-4886-a724-79efb3ddbcd7)

Data Info

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/358fdc9d-9891-4df8-a655-3abde2c14d27)

Null Dataset

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/6fa5856e-414c-41a5-b421-b4f825f0c567)

Value Count

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/795262a4-4a9f-420a-8238-86862cd5fdeb)

Salary Data

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/1728c8c5-ba32-4b0c-8f2b-55bc4a60a68f)

X head

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/fadf1d19-b1ee-41eb-ad09-7d3215a753a0)

Accuracy

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/5a2f2bf2-cbd8-4dd1-aced-c3e06ae32c65)

Prediction

![image](https://github.com/preethi2831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/155142246/fc8d9985-c207-4315-b63a-71c329d3519c)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
