import numpy as np

def confusion_matrix(true,pred):
	tp=0
	tn=0
	fp=0
	fn=0
	for i in range(0,len(true)):
		if true[i]>0.5 and pred[i]>0.5:
			tp+=1
		elif true[i]<0.5 and pred[i]<0.5:
			tn+=1
		elif true[i]>0.5 and pred[i]<0.5:
			fn+=1
		elif true[i]<0.5 and pred[i]>0.5:
			fp+=1
	return tp,tn,fp,fn

res=np.load('defense_fairness.npz')

print('Gender = 0')
print('Size =',(res['x'][:,4]<0.5).sum())
true=res['l'][res['x'][:,4]<0.5]
pred=res['z'][res['x'][:,4]<0.5]
pred_=res['z_'][res['x'][:,4]<0.5]
tp,tn,fp,fn=confusion_matrix(true=true,pred=pred)
print(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp),tp/(tp+fn))
tp,tn,fp,fn=confusion_matrix(true=true,pred=pred_)
print(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp),tp/(tp+fn))

print('Gender = 1')
print('Size =',(res['x'][:,4]>0.5).sum())
true=res['l'][res['x'][:,4]>0.5]
pred=res['z'][res['x'][:,4]>0.5]
pred_=res['z_'][res['x'][:,4]>0.5]
tp,tn,fp,fn=confusion_matrix(true=true,pred=pred)
print(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp),tp/(tp+fn))
tp,tn,fp,fn=confusion_matrix(true=true,pred=pred_)
print(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp),tp/(tp+fn))

print('Overall')

true=res['l']
pred=res['z']
pred_=res['z_']
tp,tn,fp,fn=confusion_matrix(true=true,pred=pred)
print(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp),tp/(tp+fn))
tp,tn,fp,fn=confusion_matrix(true=true,pred=pred_)
print(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp),tp/(tp+fn))