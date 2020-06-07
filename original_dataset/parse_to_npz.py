import json,os
import numpy as np
from random import shuffle
from sklearn.preprocessing import OneHotEncoder

def get_data(fname,sen_indices=[]):
	X=[]
	S=[]
	y=[]
	f=open(fname)
	rows=f.readlines()
	f.close()

	shuffle(rows)

	for row in rows:
		row=eval('['+row+']')
		y.append(row[-1])
		row=row[:-1]
		xx=[]
		ss=[]
		for i in range(0,len(row)):
			if i in sen_indices:
				ss.append(row[i])
			else:
				xx.append(row[i])
		X.append(xx)
		S.append(ss)
	f.close()

	return X,S,y

if __name__=='__main__':

	fnames=os.listdir('.')

	for fname in fnames:

		if '.csv' not in fname:
			continue
		if 'adult' in fname:
			indices=[8,9]
		if 'broward' in fname:
			indices=[0,1]
		if 'compas' in fname:
			indices=[2,1]
		if 'hospital' in fname:
			indices=[10,2]

		print(fname)

		X,S,y=get_data(fname,sen_indices=indices)

		enc=OneHotEncoder(drop='if_binary')
		X=enc.fit_transform(X).toarray()
		# print(enc.categories_)
		S=enc.fit_transform(S).toarray()
		Y=np.array(y).reshape(-1,1)
		Y=enc.fit_transform(Y).toarray().astype(int)

		#BEGIN: reweighing for race
		stat={}
		weight={}
		for i in range(0,2):
			for j in range(0,max(y)+1):
				stat[(i,j)]=0
		for i in range(0,len(y)):
			stat[(int(S[i][0]),y[i])]+=1
		for i in range(0,2):
			for j in range(0,max(y)+1):
				up=(stat[(i,0)]+stat[(i,1)])*(stat[(0,j)]+stat[(1,j)])
				down=len(y)*stat[(i,j)]
				if down==0:
					weight[(i,j)]=0
				else:
					weight[(i,j)]=up/down
		print(weight)
		reweigh_race=[]
		for i in range(0,len(y)):
			reweigh_race.append(weight[(int(S[i][0]),y[i])])
		#END: reweighing for race

		#BEGIN: reweighing for gender
		stat={}
		weight={}
		for i in range(0,2):
			for j in range(0,max(y)+1):
				stat[(i,j)]=0
		for i in range(0,len(y)):
			stat[(int(S[i][1]),y[i])]+=1
		for i in range(0,2):
			for j in range(0,max(y)+1):
				up=(stat[(i,0)]+stat[(i,1)])*(stat[(0,j)]+stat[(1,j)])
				down=len(y)*stat[(i,j)]
				if down==0:
					weight[(i,j)]=0
				else:
					weight[(i,j)]=up/down
		print(weight)
		reweigh_gender=[]
		for i in range(0,len(y)):
			reweigh_gender.append(weight[(int(S[i][1]),y[i])])
		#END: reweighing for gender

		np.savez('./npz/'+fname.split('.')[0]+'.npz',X=X,S=S,Y=Y,reweigh_race=reweigh_race,reweigh_gender=reweigh_gender)