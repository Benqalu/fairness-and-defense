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

	return np.array(X),np.array(S),np.array(y)

if __name__=='__main__':

	fnames=os.listdir('.')

	base_fnames=['adult_race-2group-white','broward_reverse','compas_reverse','hospital_race-2group-white']

	for fname in fnames:

		if fname.split('.')[0] in base_fnames:
			continue

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
		base_fname=fname.split('_')[:-2]
		base_fname='_'.join(base_fname)+'.csv'

		X,S,y=get_data(fname,sen_indices=indices)
		pickn=len(X)
		XX,SS,yy=get_data(base_fname,sen_indices=indices)

		X=np.vstack([X,XX])
		S=np.vstack([S,SS])
		y=np.hstack([y,yy])
		print(len(X),len(S),len(y))

		enc=OneHotEncoder(drop='if_binary')
		X=enc.fit_transform(X).toarray()
		# print(enc.categories_)
		S=enc.fit_transform(S).toarray()
		Y=np.array(y).reshape(-1,1)
		Y=enc.fit_transform(Y).toarray().astype(int)

		X=X[:pickn,:]
		S=S[:pickn,:]
		Y=Y[:pickn,:]
		y=y[:pickn]

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