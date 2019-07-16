
def readFile(path):
    file = open(path,'r')
    data = [line.split() for line in file]
    for row in data:
        for i in range(len(row)):
                row[i] = float(row[i])
    return data

data = readFile('Dataset')

gamma = 0.000001
epsilon = 0.000001

def split(data):
    train = []
    test = []
    for i in range(0,len(data),10):
        test.append(data[i])
        for j in range(i+1,i+10):
            if j < len(data):
                train.append(data[j])
    return train,test

train , test = split(data)

def f(w,row):
    fx = w[0]
    for i in range(len(row)-1):
        fx += w[i+1]*row[i]
    return fx

def MSE(set,w):
    E = 0
    for row in set:
        E += (row[len(row)-1]-f(w,row))**2
    return E/len(set)

def Gradient(w,row):
    G = [0]*14
    for i in range(len(G)):
        if i == 0:
            G[i] = -2*(row[len(row)-1]-f(w,row))
        else:
            G[i] = -2*row[i-1]*(row[len(row)-1]-f(w,row))
    return G

def isConstant(w, w_new):
    for i in range(len(w)):
        if abs(w_new[i] - w[i]) > epsilon:
            return False
    return True

def SGD():
    w = [0]*14
    E = 0
    for i in range(len(train)):
        E = MSE(train,w)
        for i in range(len(train)):
            G = Gradient(w,train[i])
            w_new = []
            for j in range(len(w)):
                w_new.append(w[j] - gamma*G[j])
            if isConstant(w,w_new) == True:
                return w_new,E
            w = w_new
    return w,E

w,E = SGD()
print('W = ',w)
print('MSE (train) = ',E)
print('MSE (test)  = ',MSE(test,w))