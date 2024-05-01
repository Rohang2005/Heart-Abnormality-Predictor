import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def mach_learn(data):
    reqdata=pd.read_csv('Heart_Disease_Prediction.csv')
    print(reqdata['Heart Disease'].value_counts())
    #print(reqdata.shape)
    
    splidata1=reqdata.drop(columns='Heart Disease',axis=1)
    splidata2=reqdata['Heart Disease']
    #print(splidata1)
    #print(splidata2)
    
    splidata1_train,splidata1_test,splidata2_train,splidata2_test=train_test_split(splidata1,splidata2,test_size=0.2,stratify=splidata2,random_state=2)
    #print(splidata1.shape,splidata1_train.shape,splidata1_test.shape)
    
    modreg=LogisticRegression()
    modreg.fit(splidata1_train,splidata2_train)
    splidata1_train_prediction=modreg.predict(splidata1_train)
    train_accuracy=accuracy_score(splidata1_train_prediction,splidata2_train)
    print('Accuracy on training data: ',train_accuracy)
    splidata1_test_prediction=modreg.predict(splidata1_test)
    test_accuracy=accuracy_score(splidata1_test_prediction,splidata2_test)
    print('Accuracy on test data: ',test_accuracy)
        
    input_numarray=np.asarray(data)
    input_reshape=input_numarray.reshape(1,-1)
    prediction=modreg.predict(input_reshape)
    #print(prediction)
    if(prediction[0]=='Presence'):
        print("abnormality found consult a doctor immediately")
    else:
        print("Perfectly normal heart")
def main():
    input_data=[]
    for i in range(13):
        user_input=int(input('Enter the required data(Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium):\n'))
        input_data.append(user_input)
    mach_learn(input_data)
    
if __name__=='__main__':
    main()
