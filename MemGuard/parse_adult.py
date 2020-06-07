import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from random import shuffle
import argparse

def get_data(fname):
	vec=[]
	f=open(fname)
	for row in f:
		row=eval('['+row+']')
		
		t=row[0]
		row[0]=row[8]
		row[8]=t

		t=row[1]
		row[1]=row[9]
		row[9]=t

		vec.append((row[:-1],row[-1]))
	f.close()
	return vec

if __name__=='__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', type=str,
						default='new_adult_data2-all.csv')

	parser.add_argument('--sensitive_attr', type=str,
						default='race')

	parser.add_argument('--reweighing', type=str,
						default=None)

	parser.add_argument('--unit', type=int,
						default=1000)

	args = parser.parse_args()

	vec=get_data('./dataset/'+args.input_file)
	shuffle(vec)

	if args.reweighing!=None:
		vec_reweighing=get_data('./dataset/'+args.reweighing)
		for i in range(0,args.unit):
			vec[i]=vec_reweighing[i]

	vec=vec[:5000]

	X=[]
	y=[]
	for i in range(0,len(vec)):
		item=vec[i]
		X.append(item[0])
		y.append(item[1])

	enc=OneHotEncoder(handle_unknown='ignore')
	enc.fit(X)
	X=enc.transform(X).toarray()

	if args.sensitive_attr=='race':
		s=0
		t=3
	if args.sensitive_attr=='gender':
		s=3
		t=5

	for i in range(0,len(X)):
		for j in range(s,t):
			X[i][j]=0.0

	np.savez('./data/location/data_complete.npz',x=X,y=y)
