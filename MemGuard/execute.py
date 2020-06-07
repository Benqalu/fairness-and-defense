'''
This script is used to run the the pipeline of MemGuard. 
'''
import os,sys,hashlib
import configparser
from time import sleep,time
from random import uniform

def md5(s):
	a=hashlib.md5()
	a.update(str.encode(str(s)))
	return a.hexdigest()

config = configparser.ConfigParser()
config.read('config.ini')
result_folder="../result/location/code_publish/"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
config["location"]["result_folder"]=result_folder
with open("config.ini",'w') as configfile:
    config.write(configfile)
    configfile.close()

fnames=os.listdir('../npz_dataset')

import subprocess
def exec_cmd(cmd):
	os.system(cmd)
	sleep(3)

fnames.sort(reverse=True)

print(fnames)


def clean_cache():
	os.system('rm ./data/location/data_complete.npz')
	os.system('rm ./result/location/code_publish/models/*.npz')
	os.system('rm ./result/location/code_publish/attack/*.npz')

for i in range(0,100):

	for fname in fnames:

		if '.npz' not in fname:
			continue
		if 'test' in fname:
			continue

		clean_cache()

		prefix=fname.split('.')[0]

		if not os.path.exists('./result/%s'%prefix):
			os.mkdir('./result/%s'%prefix)

		cmd='python generate_data.py --file %s'%(fname)
		exec_cmd(cmd)
		cmd="python train_user_classification_model.py -dataset location"
		exec_cmd(cmd)
		cmd="python train_defense_model_defensemodel.py  -dataset location"
		exec_cmd(cmd)
		cmd="python defense_framework.py -dataset location -qt evaluation" 
		exec_cmd(cmd)
		cmd="python train_attack_shadow_model.py -dataset location -adv adv1"
		exec_cmd(cmd)
		cmd="python evaluate_nn_attack.py -dataset location -scenario full -version v0 -outfile ./result/%s/%s.npz"%(prefix,md5(str(time())+str(uniform(0,1))))
		exec_cmd(cmd)
