import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

logo_path="img_1.png"
st.image(logo_path,width=350)

logo_path="img1.jpeg"
st.sidebar.image(logo_path,width=150)

data=pd.read_csv('online_fraud.csv')
st.sidebar.markdown('<h1 style="color:palevioletred;">SHAPE OF PAYMENT DATA IS :</h1>',unsafe_allow_html=True)
st.sidebar.write(data.shape)
st.sidebar.write(data.head(5))

#st.sidebar.write(data.columns[data.isna().any()])
st.sidebar.markdown('<h1 style="color:lightpink;">DATASET FULL DETAILS :</h1>',unsafe_allow_html=True)
data.info()
data.describe()

data['type'] = data['type'].map({'PAYMENT':0,'TRANSFER':1,'CASH_OUT':2,'DEBIT':3,'CASH_IN':4}).astype(int)

x1=data.iloc[:,[1,2,3,4,5,6]]#.values
st.sidebar.write(x1)
st.sidebar.write("shape of x is:\n",x1.shape)

y1=data.iloc[:,-1]#.values
st.sidebar.write(y1)
st.sidebar.write("y values:",y1.shape)

# Count the number of transactions for each type
transaction_counts = data['type'].value_counts()

st.sidebar.markdown('<h1 style="color:lightpink;">BAR PLOT FOR TYPE OF PAYMENTS :</h1>',unsafe_allow_html=True)
# Create a bar plot
fig_0,ax=plt.subplots(figsize=(10, 6))
ax.bar(transaction_counts.index, transaction_counts.values)

#create scatter plot
fig_1,ax=plt.subplots(figsize=(10, 6))
ax.scatter(transaction_counts.index, transaction_counts.values,color='red',marker='*')

#creat plot
fig_2,ax=plt.subplots(figsize=(10, 6))
ax.plot(range(1,10),color='blue',linestyle='-',marker='*',markerfacecolor='red',markersize=10)

ax.set_xlabel('Transaction Type')
ax.set_ylabel('Count')
ax.set_title('Transaction Type Distribution')
plt.xticks(rotation=90)
payment_types = ['Payment','Transfer','Debit','Cash Out','Cash In']
plt.xticks(range(len(payment_types)), payment_types)
st.sidebar.pyplot(fig_0)
st.sidebar.markdown('<h1 style="color:lightblue;">SCATTER PLOT FOR TYPE OF PAYMENTS :</h1>',unsafe_allow_html=True)
st.sidebar.pyplot(fig_1)
st.sidebar.markdown('<h1 style="color:lightgreen;">LINE PLOT FOR TYPE OF PAYMENTS :</h1>',unsafe_allow_html=True)
st.sidebar.pyplot(fig_2)
#adding linearRegression code
#model=LinearRegression()
#model.fit(x1,y1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.02, random_state=10)
st.sidebar.write(x_train.shape)
st.sidebar.write(y_test.shape)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#svm
from sklearn import svm
model=svm.SVC(kernel='linear') # Linear Kernel
model.fit(x_train,y_train)

#logistic
from sklearn.linear_model import (LogisticRegression)
model=LogisticRegression(random_state=0)
model.fit(x_train,y_train)

st.markdown('<h1 style="color:cyan;">ENTER YOUR PAYMENT DETAILS :</h1>',unsafe_allow_html=True)
typ=st.number_input("Enter yours Payment type:")
amt=st.number_input("Enter the amount:")
old_balance_Org=st.number_input("enter the oldbalanceOrg:")
new_balance_Org=st.number_input("enter the newbalanceOrg:")
old_balance_Dest=st.number_input("enter the oldbalanceDest:")
new_balance_Dest=st.number_input("enter the newbalanceDest:")

if (st.button("Submit")):
    input_features = [[typ,amt,old_balance_Org,new_balance_Org,old_balance_Dest,new_balance_Dest]]
    predicted_status = model.predict(input_features)
    st.sidebar.write("Prediction status of payment",round(predicted_status[0],2))

a=[[typ,amt,old_balance_Org,new_balance_Org,old_balance_Dest,new_balance_Dest]]

#st.warning("LOGISTIC REGRESSION PREDICTION RESULT")
st.markdown('<h1 style="font-size:34px;color:purple;">LOGISTIC REGRESSION PREDICTION RESULT :</h1>',unsafe_allow_html=True)


#logisticRegression code
result=model.predict(sc.transform(a))
st.write(result)

if result==1:
    st.error(f"IS FRAUD")
else:
    st.success(f"IS NOT FRAUD")

y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score
# cm=confusion_matrix(y_test,y_pred)
#
# print("confusion matrix:",cm)

st.warning("Accuracy of the model:{0}%".format(accuracy_score(y_test,y_pred)*100))
st.warning("Precession Score:{0}%".format(precision_score(y_test,y_pred,average = 'micro',zero_division = 0)*100))
st.warning("Recall Score:{0}%".format(recall_score(y_test,y_pred,average = 'micro', zero_division = 0)*100))
st.warning("f1 Score:{0}%".format(f1_score(y_test,y_pred,average = 'micro', zero_division = 0)*100))

st.markdown('<h1 style="font-size:35px;color:brown;">SVM PREDICTION RESULT :</h1>',unsafe_allow_html=True)

result=model.predict(sc.transform(a))
st.write(result)

if result==1:
    st.error(f"IS FRAUD")
else:
    st.success(f"IS NOT FRAUD")

y_pred=model.predict(x_test)

#confussion matrix to find accuracy of the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)

st.sidebar.markdown('<h1 style="color:gold;">CONFUSION MATRIX :</h1>',unsafe_allow_html=True)
st.sidebar.write(cm)

st.warning(" Accuracy of the model:{0}%".format(accuracy_score(y_test,y_pred)*100))

if result==1:
    logo_path = "img.png"
    st.image(logo_path, width=400)
else:
    logo_path = "img_2.png"
    st.image(logo_path, width=400)