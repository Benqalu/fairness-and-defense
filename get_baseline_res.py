import numpy as np 

data={}
f=open('result_nn_aif360_horizental.txt')
for row in f:
	row=row.split('\t')
	key=' '.join(row[:-1])
	value=float(row[-1])
	if key not in data:
		data[key]=[]
	data[key].append(value)
f.close()

for item in data:
	if len(data[item])==0:
		continue
	print(item,len(data[item]),float(np.mean(np.abs(data[item]))))

'''

for item in sorted(data.keys()):
	if type(data[item])!=float:
		continue
	print(item,data[item])

'''