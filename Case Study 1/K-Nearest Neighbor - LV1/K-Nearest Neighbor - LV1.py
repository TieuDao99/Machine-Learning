import numpy as np 
from random import randint

#đọc dữ liệu
def readFile(filepath):		
	save = []
	with open(filepath, "r") as fi:
		while True:
			line = fi.readline()
			if (not line):
				break
			tmp = line.strip().split(' ')
			save.append([i for i in tmp if i != ''])
	return save

# bảng ánh xạ dữ liệu
social_class = {'1st' : '0', '2nd' : '1', '3rd' : '2', 'crew' : '3'}
age = {'child' : '0', 'adult' : '1'}
sex = {'male' : '0', 'female' : '1'}

# tiếp cận 1, biến đổi dữ liệu
def convert(data):	
	save = []
	for i in range(len(data)):
		save.append([social_class[data[i][0]] + age[data[i][1]] + sex[data[i][2]] + " " + data[i][3]])
	return save

dataset = readFile("Dataset.data")
dataset = convert(dataset)

def split2(data):
	
	save_training, save_test = [], []
	colection = [[], [], [], []]
	good = [True] * len(data)
	for i in range(len(data)):
		colection[int(data[i][0][0])].append(i)
	
	for i in range(4):
		s = set()
		while (len(s) < 10):
			s.update([randint(0, len(colection[i]))])
		for j in s:
			good[j] = False
			save_test.append(data[colection[i][j]])

	tmpdata = []
	for i in range(len(data)):
		if (good[i]):
			tmpdata.append(data[i])

	value, cnt = np.unique(np.array(tmpdata)[:, 0], return_counts = True)

	for i in range(len(value)):
		save_training.append([value[i], cnt[i]])
	
	return [save_training, save_test]

training_set, test_set = split2(dataset)

def dist(X, Y):
	return abs(int(X[0]) - int(Y[0])) + abs(int(X[1]) - int(Y[1])) + abs(int(X[2]) - int(Y[2]))

def cnt(X, k):

	tmp = []
	for i in range(len(training_set)): 
		tmp.append([dist(training_set[i][0][0:3], X), i])
	
	for i in range(len(tmp)):
		for j in range(i + 1, len(tmp)):
			if (tmp[i][0] > tmp[j][0]):
				temp = tmp[i]
				tmp[i] = tmp[j]
				tmp[j] = temp
	
	cntYes, cntNo = 0, 0
	i = 0
	kk = k
	check = set()
	while (kk > 0 and i < len(tmp)):
		d = training_set[tmp[i][1]]
		if (d[0][0:3] not in check):
			kk -= 1
			check.update([d[0][0:3]])
		if (d[0][4] == 'n'):
			cntNo += int(d[1])
		else:
			cntYes += int(d[1])
		i += 1
	return [cntYes, cntNo]

k = max(min(16, int(input("Nhập K: "))), 0)

score = 0

for i in test_set:
	cYes, cNo = cnt(i[0][0: 3], k)
	ans = "n"
	if (cYes >= cNo):
		ans = "y"

	if (ans == i[0][4]):
		score += 1

print(round(score / 40.0 * 100, 2), "%")

