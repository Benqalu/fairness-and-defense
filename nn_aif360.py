import os,keras,hashlib
import numpy as np
from copy import deepcopy
from keras import backend as K
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
from sklearn.utils import shuffle
from time import time
from random import uniform

def md5(s):
	a=hashlib.md5()
	a.update(str.encode(str(s)))
	return a.hexdigest()

def target_model(input_shape,labels_dim):
	inputs=Input(shape=input_shape)
	middle_layer=Dense(1024,activation='relu')(inputs)
	middle_layer=Dense(512,activation='relu')(middle_layer)
	middle_layer=Dense(256,activation='relu')(middle_layer)
	middle_layer=Dense(128,activation='relu')(middle_layer)
	outputs_logits=Dense(labels_dim)(middle_layer)
	outputs=Activation('softmax')(outputs_logits)
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
	model.summary()
	return model

def mia_model(input_shape,labels_dim):
	inputs_b=Input(shape=input_shape)
	x_b=Dense(512,activation='relu')(inputs_b)
	x_b=Dense(256,activation='relu')(x_b)
	x_b=Dense(128,activation='relu')(x_b)
	outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform())(x_b)
	outputs=Activation('sigmoid')(outputs_pre)
	model = Model(inputs=inputs_b, outputs=outputs)
	model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
	model.summary()
	return model

def aif_reweigh(X,S,Y,w):
	n=len(X)
	w=w/w.sum()

	fair_X=[]
	fair_S=[]
	fair_Y=[]

	train_X=[]
	train_S=[]
	train_Y=[]

	indices=[i for i in range(0,n)]
	selected=[]
	for i in range(0,n):
		z=np.random.choice(indices,p=w)
		selected.append(z)
	print('Selected size = %d, actual size = %d'%(len(selected),len(set(selected))))

	fair_X=X[selected]
	fair_S=S[selected]
	fair_Y=Y[selected]

	selected=list(set(selected))

	train_X=X[selected]
	train_S=S[selected]
	train_Y=Y[selected]

	return fair_X,fair_S,fair_Y,train_X,train_S,train_Y

def get_result(basename,attribute,modification='original'):

	print(basename,attribute,modification)

	a=np.load('./npz_dataset/'+basename+'_train_.npz')
	train_X=a['X']
	train_S=a['S']
	train_Y=a['Y']
	train_w=a['reweigh_%s'%attribute]
	train_X,train_S,train_Y,train_w=shuffle(train_X,train_S,train_Y,train_w)

	a=np.load('./npz_dataset/'+basename+'_test_.npz')
	test_X=a['X']
	test_S=a['S']
	test_Y=a['Y']
	train_X,train_S,train_Y=shuffle(train_X,train_S,train_Y)

	#BEGIN: Get fair dataset
	if modification=='original':
		fair_X=deepcopy(train_X)
		fair_S=deepcopy(train_S)
		fair_Y=deepcopy(train_Y)
	elif modification=='aif360':
		fair_X,fair_S,fair_Y,train_X,train_S,train_Y=aif_reweigh(X=train_X,S=train_S,Y=train_Y,w=train_w)
	elif 'fairpick' in modification:
		degree=modification.split('_')[-1]
		a=np.load('./npz_dataset/'+basename+'_%s_%s.npz'%(attribute,degree))
		print('./npz_dataset/'+basename+'_%s_%s.npz'%(attribute,degree))
		fair_X=a['X']
		fair_S=a['S']
		fair_Y=a['Y']
		fair_X,fair_S,fair_Y=shuffle(fair_X,fair_S,fair_Y)
	else:
		exit()

	pickn=min(len(fair_X),len(train_X))
	pickn=min(pickn,len(test_X))
	train_X=train_X[:pickn,:]
	train_S=train_S[:pickn,:]
	train_Y=train_Y[:pickn,:]
	test_X=test_X[:pickn,:]
	test_S=test_S[:pickn,:]
	test_Y=test_Y[:pickn,:]
	print('Size =',pickn)
	#END: Get fair dataset

	#BEGIN: train target model
	fair_X=np.hstack([fair_X,fair_S])
	if len(fair_Y[0])==1:
		fair_Y=keras.utils.to_categorical(fair_Y)
	target=target_model(input_shape=fair_X[0].shape,labels_dim=len(fair_Y[0]))
	epochs=100
	batch_size=64
	index_array=np.arange(fair_X.shape[0])
	batch_num=np.int(np.ceil(fair_X.shape[0]/batch_size))
	for i in np.arange(epochs):
		np.random.shuffle(index_array)
		for j in np.arange(batch_num):
			b_batch=fair_X[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,fair_X.shape[0])],:]
			y_batch=fair_Y[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,fair_Y.shape[0])]]
			target.train_on_batch(b_batch,y_batch)   
		if (i+1)%10==0:
			K.set_value(target.optimizer.lr,K.eval(target.optimizer.lr*0.8))
			print("Epochs: {}".format(i+1))
			scores_train=target.evaluate(fair_X,fair_Y,verbose=0)
			print('FairTrain loss:', scores_train[0])
			print('FairTrain accuracy:', scores_train[1])
	fair_Y_pred=target.predict(fair_X)
	#END: train target model

	#BEGIN: get predicted result
	train_X=np.hstack([train_X,train_S])
	train_Y_pred=target.predict(train_X)
	test_X=np.hstack([test_X,test_S])
	test_Y_pred=target.predict(test_X)
	print(train_Y_pred)
	print(test_Y_pred)
	#END: get predicted result

	#BEGIN: Create MIA data
	n=len(train_X)
	train_MIA_X=np.vstack([train_Y_pred[:n//2,:],test_Y_pred[:n//2,:]])
	train_MIA_y=np.hstack([np.ones(n//2),np.zeros(n//2)])
	test_MIA_X=np.vstack([train_Y_pred[n//2:n//2+n//2,:],test_Y_pred[n//2:n//2+n//2,:]])
	test_MIA_target_X=np.vstack([train_X[n//2:n//2+n//2,:],test_X[n//2:n//2+n//2,:]])
	test_MIA_y=np.hstack([np.ones(n//2),np.zeros(n//2)])
	#END: Create MIA data

	#BEGIN: train MIA
	# train_MIA_y=keras.utils.to_categorical(train_MIA_y)
	# test_MIA_y=keras.utils.to_categorical(test_MIA_y)
	mia=mia_model(input_shape=train_MIA_X[0].shape,labels_dim=1)
	epochs=100
	batch_size=64
	index_array=np.arange(train_MIA_X.shape[0])
	batch_num=np.int(np.ceil(train_MIA_X.shape[0]/batch_size))
	for i in np.arange(epochs):
		np.random.shuffle(index_array)
		for j in np.arange(batch_num):
			b_batch=train_MIA_X[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,train_MIA_X.shape[0])],:]
			y_batch=train_MIA_y[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,train_MIA_y.shape[0])]]
			mia.train_on_batch(b_batch,y_batch)   
		if (i+1)%10==0:
			K.set_value(mia.optimizer.lr,K.eval(mia.optimizer.lr*0.8))
			print("Epochs: {}".format(i+1))
			scores_train = mia.evaluate(train_MIA_X, train_MIA_y, verbose=0)
			print('Train loss:', scores_train[0])
			print('Train accuracy:', scores_train[1])
	train_MIA_y_pred=mia.predict(train_MIA_X)
	#END: train MIA

	#BEGIN: get predicted result
	test_MIA_y_pred=mia.predict(test_MIA_X)
	print(test_MIA_y_pred)
	#END: get predicted result

	#BEGIN: Calculate VD
	test_MIA_y_pred=(test_MIA_y_pred.reshape(-1)>0.5).astype(int)
	print(test_MIA_y_pred)

	if attribute=='race':
		print('Race:',end='')
		tmp_y_true_0=test_MIA_y[test_MIA_target_X[:,-2]==0]
		tmp_y_pred_0=test_MIA_y_pred[test_MIA_target_X[:,-2]==0]
		recall_0=(tmp_y_true_0*tmp_y_pred_0==1).sum()/(tmp_y_true_0==1).sum()
		tmp_y_true_1=test_MIA_y[test_MIA_target_X[:,-2]==1]
		tmp_y_pred_1=test_MIA_y_pred[test_MIA_target_X[:,-2]==1]
		recall_1=(tmp_y_true_1*tmp_y_pred_1==1).sum()/(tmp_y_true_1==1).sum()
		VD=recall_0-recall_1
		print(VD)
	if attribute=='gender':
		print('Gender:',end='')
		tmp_y_true_0=test_MIA_y[test_MIA_target_X[:,-1]==0]
		tmp_y_pred_0=test_MIA_y_pred[test_MIA_target_X[:,-1]==0]
		recall_0=(tmp_y_true_0*tmp_y_pred_0==1).sum()/(tmp_y_true_0==1).sum()
		tmp_y_true_1=test_MIA_y[test_MIA_target_X[:,-1]==1]
		tmp_y_pred_1=test_MIA_y_pred[test_MIA_target_X[:,-1]==1]
		recall_1=(tmp_y_true_1*tmp_y_pred_1==1).sum()/(tmp_y_true_1==1).sum()
		VD=recall_0-recall_1
		print(VD)
	#END: Calculate VD

	np.savez('./raw_result/%s_%s_%s_%s.npz'%(basename,attribute,modification,md5(str(time())+str(uniform(0,1)))),
		target_train_Y=fair_Y,
		target_train_Y_pred=fair_Y_pred,
		target_test_Y=test_Y,
		target_test_Y_pred=test_Y_pred,
		mia_train_y=train_MIA_y,
		mia_train_y_pred=train_MIA_y_pred,
		mia_test_y=test_MIA_y,
		mia_test_y_pred=test_MIA_y_pred,
		VD=VD
		)

	return VD

if __name__=='__main__':
	base_fnames=['adult_race-2group-white','broward_reverse','compas_reverse','hospital_race-2group-white']
	settings=['original','aif360','fairpick_80','fairpick_60','fairpick_40']

	fnames=os.listdir('./npz_dataset/')
	for _ in range(0,20):
		for fname in base_fnames:
			for attribute in ['race','gender']:
				for setting in settings:

					if setting!='original':
						continue

					print(fname,setting)
					f=open('result_nn_aif360.txt','a')
					f.write('%s\t%s\t%s\n'%(fname,attribute,setting))
					VD=get_result(basename=fname,attribute=attribute,modification=setting)
					f.write(str(VD)+'\n')
					f.close()
