def readFile(path):
    arr=[]
    file = open(path,'r',encoding='utf-8')
    for line in file:
        data = line.strip()
        a = data.split()
        arr.append(a)
    file.close()
    return arr

t = readFile('TitanicDataset')
print('-----INPUT DATASET: {0}'.format(len(t)))
print('DATASET: ',t)
def convert(t):
    for i in range(len(t)):
        if t[i][0] == '1st': t[i][0]=0
        elif t[i][0] == '2nd': t[i][0]=1
        elif t[i][0] == '3rd': t[i][0]=2
        else: t[i][0] = 3

        if t[i][1] == 'adult': t[i][1]=0
        else: t[i][1]=1

        if t[i][2] == 'male': t[i][2]=0
        else: t[i][2] = 1

        if t[i][3] == 'yes': t[i][3]=1
        else: t[i][3]=0
    return t

convert(t)
print('------>:  ',t)

print('\n-----SPLIT DATASET INTO DATA AND LABEL:')
def split(t):
    data = []
    label = []
    for i in range(len(t)):
        data.append([t[i][0],t[i][1],t[i][2]])
        label.append(t[i][3])
    return data,label

data,label = split(t)
print('DATA: ',len(data),data)
print('LABEL: ',len(label),label)

print('\n-----SPLIT DATA AND LABEL INTO DATA(TRAIN, TEST) AND LABEL(TRAIN, TEST):')
from sklearn.model_selection import train_test_split

data_train,data_test,label_train,label_test = train_test_split(data,label,test_size=0.1,random_state=1)
print('DATA TRAIN: ',len(data_train),data_train)
print('LABEL TRAIN: ',len(label_train),label_train)
print('DATA TEST: ',len(data_test),data_test)
print('LABEL TEST: ',len(label_test),label_test)

print('\n-----K - NEAREST NEIGHBOR ...')
from sklearn import neighbors
NB1 = neighbors.KNeighborsClassifier(n_neighbors=1) #default weight ='uniform', p(norm)=2
NB1.fit(data_train,label_train)

NB10 = neighbors.KNeighborsClassifier(n_neighbors=10) #increase k to 10
NB10.fit(data_train,label_train)

NB10_dt = neighbors.KNeighborsClassifier(n_neighbors=10,weights='distance') #weight = 'distance'
NB10_dt.fit(data_train,label_train)

import numpy as np
def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

NB10_dt_mw = neighbors.KNeighborsClassifier(n_neighbors=10,weights=myweight) #weight = myweight
NB10_dt_mw.fit(data_train,label_train)

predicted_label_1 = NB1.predict(data_test)
predicted_label_10 = NB10.predict(data_test)
predicted_label_10_dt = NB10_dt.predict(data_test)
predicted_label_10_dt_mw = NB10_dt_mw.predict(data_test)

print('done !\n')
print('PREDICTED LABEL TEST k = 1: ',len(predicted_label_1),'\n',predicted_label_1)
print('PREDICTED LABEL TEST k =10: ',len(predicted_label_10),'\n',predicted_label_10)
print('PREDICTED LABEL TEST k =10 - weight = distance: ',len(predicted_label_10_dt),'\n',predicted_label_10_dt)
print('PREDICTED LABEL TEST k =10 - weight = myweight: ',len(predicted_label_10_dt_mw),'\n',predicted_label_10_dt_mw)

from sklearn.metrics import accuracy_score
print('\n- accuracy = %.2f%%'%(accuracy_score(label_test,predicted_label_1)*100))
print('\n- accuracy = %.2f%%'%(accuracy_score(label_test,predicted_label_10)*100))
print('\n- accuracy = %.2f%%'%(accuracy_score(label_test,predicted_label_10_dt)*100))
print('\n- accuracy = %.2f%%'%(accuracy_score(label_test,predicted_label_10_dt_mw)*100))