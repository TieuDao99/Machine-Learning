
import numpy as np 

def inputData(filename):
	save = []
	with open(filename) as fi:
		while True:
			line = fi.readline()
			if (not line): 
				break
			line = line.strip().split(' ')
			tmp = [i  for i in line if i != '']
			save.append(tmp)
	return save

data = np.array(inputData("Dataset.data"))

print("---------Input------------")
for i in range(data.shape[1] - 1):
	value, cnt = np.unique(data[:, i], return_counts = True)
	print(dict(zip(value, cnt)))
print("---------Label------------")
value, cnt = np.unique(data[:, data.shape[1] - 1], return_counts = True)
print(dict(zip(value, cnt)))