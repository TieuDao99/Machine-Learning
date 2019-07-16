import numpy as np
from sklearn.model_selection import train_test_split

def readFile(path):
    file = open(path,'r')
    data = [line.split() for line in file]
    for row in data:
        for i in range(len(row)):
                row[i] = float(row[i])
    return data
data = readFile('Dataset')

def normalize():
    for row in data:
        row.insert(0,1)
normalize()

def split():
    X = []
    y = []
    for row in data:
        temp = []
        for i in row:
            if i is row[14]:
                y.append(i)
            else:
                temp.append(i)
        X.append(temp)
    return X,y
X,y = split()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1)
X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

def NE_basic():
    A = (X_train.T).dot(X_train)
    A_inverted = np.linalg.inv(A)
    b = (X_train.T).dot(y_train)
    return A_inverted.dot(b)

w = NE_basic()

def MSE(n,y, X):
    return (np.linalg.norm(y - X.dot(w),2))**2/n

E_train = MSE(X_train.shape[0],y_train,X_train)
E_test = MSE(X_test.shape[0],y_test,X_test)

print('W = ',w)
print('MSE (train) = ',E_train)
print('MSE (test)  = ',E_test)