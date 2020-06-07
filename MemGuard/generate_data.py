import argparse
import numpy as np
from sklearn.utils import shuffle

def fetch(path,fname):

	print('Traing dataset :',fname)

	base_data='_'.join(fname.split('.')[0].split('_')[:-2])+'_train_.npz'

	attribute=fname.split('.')[0].split('_')[-2]

	print('Base Data :',base_data)
	print('Sensitive :',attribute)

	#BEGIN: Assemble training
	a=np.load(path+fname,allow_pickle=True)
	X=a['X']
	S=a['S']
	Y=a['Y']
	X,S,Y=shuffle(X,S,Y)
	X=np.hstack([S,X])
	res_X=X[:1000,:]
	res_Y=Y[:1000,:]
	#END: Assemble trainging

	#BEGIN: Assemble rest
	a=np.load(path+base_data,allow_pickle=True)
	X=a['X']
	S=a['S']
	Y=a['Y']
	X,S,Y=shuffle(X,S,Y)
	X=np.hstack([S,X])
	X=X[:4000,:]
	Y=Y[:4000,:]

	res_X=np.vstack([res_X,X])
	res_Y=np.vstack([res_Y,Y])
	#END: Assemble rest

	#BEGIN: Adjust & Save
	res_y=res_Y.reshape(-1)
	np.savez('./data/location/data_complete.npz',x=res_X,y=res_y)
	#END: Adjust & Save


if __name__=='__main__':

	p=argparse.ArgumentParser()
	p.add_argument('--file',type=str,default=None)
	p=p.parse_args()

	if p.file==None:
		exit()

	fetch(path='../npz_dataset/',fname=p.file)