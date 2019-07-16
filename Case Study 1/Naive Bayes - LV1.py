
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
print('-----INPUT DATA: {0}\n'.format(len(t)))


def statistics(t,i):
    st=[]
    for row in t:
        flag = False
        for r in st:
            if row[i] == r[0]:
                r[1]+=1
                flag =True
                break
        if flag == False:
            st.append([row[i], 1])
    return st

print('social class:     ', statistics(t,0))
print('age:              ', statistics(t,1))
print('sex:              ', statistics(t,2))
print('survived or dead: ', statistics(t,3))


print('\n-----SPLIT INTO TRAINING SET AND TEST SET - BASE ON SOCIAL CLASS ATTRIBUTE: \n')

def split(t):
    test_set = []
    social = statistics(t, 0)
    temp = 0
    for i in range(len(social)):
        for j in range(temp,temp+social[i][1],10):
            test_set.append(t[j])
        temp += social[i][1]

    train_set = t
    for i in range(len(test_set)):
        for j in range(len(train_set)):
            if train_set[j]== test_set[i]:
                train_set.remove(train_set[j])
                break
    return train_set,test_set

train_set,test_set = split(t)

print('train set: ',len(train_set))
print(train_set)
print('test set: ',len(test_set))
print(test_set)


print('\n-----STATISTICS ON TRAINING SET:\n')

def stati_yesno(data_set,i):
    st=[]
    for row in data_set:
        flag = False
        for r in st:
            if row[i] == r[0]:
                if row [3]=='yes':
                    r[1] += 1
                else:
                    r[2]+=1
                flag = True
                break
        if flag == False:
            if row[3] == 'yes':
                st.append([row[i], 1,0])
            else:
                st.append([row[i], 0,1])
    return st

sc = stati_yesno(train_set,0)
age = stati_yesno(train_set,1)
sex = stati_yesno(train_set,2)
alive = statistics(train_set,3)
print('  >   social class:     ', sc)
print('  >   age:              ', age)
print('  >   sex:              ', sex)
print('  >   survived or dead: ', alive)

def probability(attribute,alive):
    pro = []
    for i in range(len(attribute)):
        new = []
        new.append(attribute[i][0])
        new.append(attribute[i][1]/alive[0][1])
        new.append(attribute[i][2]/alive[1][1])
        pro.append(new)
    return pro


print('\n-----COMPUTE PROBABILITIES ON TRAINING SET: \n')

yes = alive[0][1]
no = alive[1][1]
survive = [['yes',yes/len(train_set)],['no',no/len(train_set)]]
pro_sc = probability(sc,alive)
pro_age = probability(age,alive)
pro_sex = probability(sex,alive)
print('  >   social class:     ', pro_sc)
print('  >   age:              ', pro_age)
print('  >   sex:              ', pro_sex)
print('  >   survived or dead: ', survive)

print('\n-----TESTING....\n')
def predict(point,pro_sc,pro_age,pro_sex,survive):
    for row in pro_sc:
        if row[0]==point[0]:
            p_x0_yes=row[1]
            p_x0_no=row[2]
            break
    for row in pro_age:
        if row[0]==point[1]:
            p_x1_yes=row[1]
            p_x1_no=row[2]
            break
    for row in pro_sex:
        if row[0]==point[2]:
            p_x2_yes=row[1]
            p_x2_no=row[2]
            break
    likelihood_yes_pyes = p_x0_yes*p_x1_yes*p_x2_yes*survive[0][1]
    likelihood_no_pno = p_x0_no*p_x1_no*p_x2_no*survive[1][1]
    p_x = likelihood_yes_pyes + likelihood_no_pno
    p_yes_x = likelihood_yes_pyes/p_x
    p_no_x = likelihood_no_pno/p_x
    if p_yes_x >= p_no_x:
        return 'yes'
    else:
        return 'no'

def test(test_set,pro_sc,pro_age,pro_sex,survive):
    for i in range(len(test_set)):
        temp = predict(test_set[i],pro_sc,pro_age,pro_sex,survive)
        if temp == test_set[i][3]:
            test_set[i].append('correctly predicted')
        else:
            test_set[i].append('incorrectly predicted')


test(test_set,pro_sc,pro_age,pro_sex,survive)
for row in test_set:
    print(row)
    print()

def accuracy(test_set):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][4]=='correctly predicted':
            correct+=1
    return correct/len(test_set)

print('accuracy score = {}%'.format(accuracy(test_set)*100))

