import os

fnames=os.listdir('./npz_dataset/')

for fname in fnames:
	if '.npz' not in fname:
		continue
	fname=fname.split('.')[0]
	for ratio in ['0.8','0.6','0.4']:
		print(fname,ratio)
		cmd='python bo_reweighing.py --ratio %s --datasetpath ./original_dataset/ --originalfile %s --outputfile %s'%(ratio,fname+'.csv',fname+'_fairpick_%s.csv'%ratio)
		os.system(cmd)