# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.       
2.Read the data frame using pandas.         
3.Get the information regarding the null values present in the dataframe.      
4.Split the data into training and testing sets.          
5.convert the text data into a numerical representation using CountVectorizer.     
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.       
7.Finally, evaluate the accuracy of the model.         

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: N.Navya Sree     
RegisterNumber: 212223040138    
*/
```
```
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy

```

## Output:
## Result Output

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/4d1aa265-56bc-4963-94f2-0ca6d0c20586)

## data.head()

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/2cfb3313-1dc2-4343-a3f4-e2db7c03f707)

## data.info()

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/8a2db0dd-50b6-44e5-a867-4194aef41d4d)

## data.isnull().sum()

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/fe8415f0-7196-4c4b-81b5-3f676b0d0cd5)

## Y_prediction Value

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/6219d833-49e8-4826-b4d8-67b1a8da0272)

## Accuracy Value

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/2bd3430d-6b6d-487e-a408-65239679430e)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
