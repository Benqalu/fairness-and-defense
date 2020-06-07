import sys,warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	warnings.filterwarnings("ignore",message="numpy.dtype size changed")
	import keras
	import numpy as np
	from keras import backend as K
	from keras.models import Model,Sequential
	from keras.layers import Dense,Dropout,Activation,Input,concatenate
	from sklearn.model_selection import train_test_split

class MemGuard():
	def __init__(self):
		self.target_data=None
		self.target_label=None
		self.target_model=None
		self.target_train_X=None
		self.target_train_Y=None
		self.target_train_Y_pred=None
		self.target_test_X=None
		self.target_test_Y=None
		self.target_test_Y_pred=None

	def set_dataset(self,data,label,n_category=None):
		self.target_data=np.array(data)
		if n_category!=None:
			self.target_label=np.array(label).astype(int)
			self.target_label-=self.target_label.min()
			self.target_label=keras.utils.to_categorical(self.target_label,n_category)
		else:
			self.target_label=np.array(label)

	def train_target(self,network,activation='relu',test_ratio=0.3,shuffle=True,batch_size=64,epochs=200):

		print('\n'*3)
		print('Target Model :')

		if network==None or len(network)<2:
			print('Target network must have at least input and output layers (2 layers).')
			exit()

		if network[0]==None:
			network[0]=len(self.target_data[0])
		print('Neural Network :',network)

		layers=[Input(shape=(network[0],))]
		for i in range(1,len(network)-1):
			layers.append(Dense(network[i],activation=activation)(layers[-1]))
		layers.append(Activation('softmax')(Dense(network[-1])(layers[-1])))

		model=Model(inputs=layers[0], outputs=layers[-1])
		model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
		model.summary()

		X_train,X_test,Y_train,Y_test=train_test_split(self.target_data,self.target_label,test_size=test_ratio,shuffle=shuffle)

		for epoch in range(epochs):
			position=0
			while position+batch_size<len(X_train):
				model.train_on_batch(X_train[position:min(position+batch_size,len(X_train))],Y_train[position:min(position+batch_size,len(Y_train))])
				position+=batch_size
			print('Training epochs: %d/%d'%(epoch+1,epochs),end='\r')
			sys.stdout.flush()
			# if (epochs+1)%150==0:
			# 	K.set_value(model.optimizer.lr,K.eval(model.optimizer.lr*0.1))
		print()

		score_train = model.evaluate(X_train,Y_train,verbose=0)
		print('Train loss:',score_train[0])
		print('Train accuracy:',score_train[1])  
		score_test = model.evaluate(X_test,Y_test,verbose=0)
		print('Test loss:',score_test[0])
		print('Test accuracy:',score_test[1])  

		self.target_model=model

		self.target_train_X=X_train
		self.target_train_Y=Y_train
		self.target_train_Y_pred=self.target_model.predict(self.target_train_X)
		self.target_test_X=X_train
		self.target_test_Y=Y_train
		self.target_test_Y_pred=self.target_model.predict(self.target_test_X)

	def train_defense(self,network,activation='relu',test_ratio=0.3,shuffle=True,batch_size=64,epochs=200):
		print('\n'*3)
		print('Defense Model :')

		if network==None or len(network)<2:
			print('Defense network must have at least input and output layers (2 layers).')
			exit()

		if network[0]==None:
			network[0]=len(self.target_data[0])
		print('Neural Network :',network)

		layers=[Input(shape=(network[0],))]
		for i in range(1,len(network)-1):
			layers.append(Dense(network[i],activation=activation)(layers[-1]))
		layers.append(Activation('sigmoid')(Dense(network[-1])(layers[-1])))

		model=Model(inputs=layers[0], outputs=layers[-1])
		model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
		model.summary()

if __name__=='__main__':

	npzdata=np.load('./data/location/data_complete.npz',allow_pickle=True)
	x_data=npzdata['x'][:,:]
	y_data=npzdata['y'][:]

	memguard=MemGuard()
	memguard.set_dataset(data=x_data,label=y_data,n_category=30)
	memguard.train_target(network=[None,1024,512,256,128,30])