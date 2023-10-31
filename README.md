# Ex 04 - Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.mport the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: arun.j
RegisterNumber:  212222040015
*/


import pandas as pd
data=pd.read_csv("/content/Placement_Data (2).csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()
data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
# Placement Data:
![Screenshot 2023-09-14 091953](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/13c83e06-ce38-45ee-b421-e3143398963c)
# Salary Data:
![Screenshot 2023-09-14 092028](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/5ac3b449-45e5-42b8-93ff-3cff53cd2142)
# Checking the null() function:
![Screenshot 2023-09-14 092042](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/58ff707e-5ab7-4a22-a06a-4c50dbe29225)
# Data Duplicate:
![Screenshot 2023-09-14 092056](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/d2c78f07-a421-4ab5-a693-40998adb1ea5)
# print Data:
![Screenshot 2023-09-14 092217](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/f4e87389-cc9e-4a33-ab3e-b9f213c02c8a)
# Data-Status
![Screenshot 2023-09-14 092247](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/e8ca9aea-cad4-434b-a380-bfc8b77e9494)
# Y_Prediction array:
![Screenshot 2023-09-14 092322](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/5faa5097-f601-448f-a283-97044713bd3f)
# Accuracy:
![Screenshot 2023-09-14 092335](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/a078773b-3353-4788-8a39-2a194edbaaa6)
# classification_report:
![Screenshot 2023-09-14 092452](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/66919176-5280-4337-bf8d-4d201c4a7ac2)

# LR_Prediction:

![Screenshot 2023-09-14 092505](https://github.com/arun1111j/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128461833/63464877-d71b-4f98-9e23-b980a78ed164)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
