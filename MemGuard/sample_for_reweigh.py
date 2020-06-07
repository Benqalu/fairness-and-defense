from random import shuffle

def func(fname):
	f=open('./dataset/'+fname)
	data=f.readlines()
	f.close()

	if data[-1].strip()=='':
		data=data[:-1]

	shuffle(data)

	f=open('./dataset/'+fname.split('.')[0]+'-for-reweigh'+'.csv','w')
	for i in range(0,2000):
		f.write(data[i].strip()+'\n')
	f.close()

if __name__=='__main__':
	func('new_adult_data2-all.csv')