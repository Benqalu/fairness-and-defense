import pandas as pd
import numpy as np
import collections
import argparse

def check_dist():
	original_data = pd.read_csv(args.inputfile, header=None, delimiter=',')
	original_data = original_data.values

	c = collections.defaultdict(int)

	for itm in original_data:
		c[itm[args.prot]] += 1

	print (c) 

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        #reweigh setttings
        parser.add_argument('--inputfile', type=str, help='reweigh ratio', default = 'dummy.csv')
        parser.add_argument('--prot', type=int, help='the sensitive attribute index', default = 9)

	args = parser.parse_args()
	check_dist()
