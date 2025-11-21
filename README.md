# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PAVITHRA S
RegisterNumber: 212223220072 
*/
Program to implement the SVM For Spam Mail Detection..
Developed by: Ramya.P
RegisterNumber: 212223240137

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.info
data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

<img width="917" height="502" alt="507403751-0a0254c8-e168-4fa9-9a5e-c0c63ce9aa6f" src="https://github.com/user-attachments/assets/4ff05911-bf32-4013-b626-26e4073cf0fa" />


Confusion matrix:
<img width="125" height="48" alt="507403880-6a534a38-30a3-42b4-883d-139e6d8d6d2e" src="https://github.com/user-attachments/assets/72d330be-c0ff-48d7-92d9-8ef38080d5bc" />


classification:

<img width="602" height="221" alt="507403129-04d05fbe-6c6c-4b6e-a7ab-2e9f31d5b0f9" src="https://github.com/user-attachments/assets/05ddf349-2531-4c47-85ed-50dc5fa63dc3" />

Accuracy:
<img width="237" height="38" alt="507404055-cabbb73e-1900-467c-b27b-9d49cdc945d0" src="https://github.com/user-attachments/assets/07816d1e-160e-4b47-98bf-30357a03cf1e" />


# RESULT:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.













