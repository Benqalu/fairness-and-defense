#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import os
import collections
from scipy.optimize import fsolve
from pulp import *
from sklearn.cluster import KMeans
from kmodes import kprototypes
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder


def solve_two_groups(x):
	global F1, F2, M1, M2, r, A, B
	out = [(F1 / A - M1 / B) * r - (F1 - x[0]) / (A - x[0]) + M1 / (B
		   - x[1])]
	out.append((M2 / B - F2 / A) * r - (M2 - x[1]) / (B - x[1]) + F2
			   / (A - x[0]))
	return out


def solve_x(x1):
	global F11, M11, A, B, Y, r
	out = (F11 / A - M11 / B) * r - (F11 - x1) / (A - X) + M11 / (B - Y)
	return out


def solve_y(y1):
	global F22, M22, A, B, X, r
	out = (M22 / B - F22 / A) * r - (M22 - y1) / (B - Y) + F22 / (A - X)
	return out


def normal_reweigh(data, original_data):

	# ####################### find distribution difference ####################

	global F1, M1, F2, M2, A, B, r
	r = args.ratio
	(
		F1,
		M1,
		F2,
		M2,
		A,
		B,
		) = (
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		)

	comb_list = set()
	for itm in data:
		tmp = ''
		for j in range(len(itm)):
			if j != args.sens_att_index and j != len(data[0]) - 1 and j \
				!= len(data[0]) - 2:
				tmp += str(itm[j]) + '-'
		comb_list.add(tmp)

	(male, female) = ({}, {})
	for itm in comb_list:
		(male[itm], female[itm]) = ([], [])

	for itm in data:
		tmp = ''
		for j in range(len(itm)):
			if j != args.sens_att_index and j != len(data[0]) - 1 and j \
				!= len(data[0]) - 2:
				tmp += str(itm[j]) + '-'
		if itm[args.sens_att_index] == 0:
			B += 1.0
			male[tmp].append(itm)
		else:
			A += 1.0
			female[tmp].append(itm)

	(male_larger_group, female_smaller_group, female_larger_group) = \
		([], [], [])
	(male_smaller_group, male_equal, female_equal) = ([], [], [])
	(not_full_cover_count, female_d_c, male_d_c) = (0, 0, 0)
	for itm in comb_list:
		if len(female[itm]) / A > len(male[itm]) / B:
			F1 += len(female[itm])
			female_larger_group.append(female[itm])
			M1 += len(male[itm])
			male_smaller_group.append(male[itm])
			female_d_c += 1
		elif len(female[itm]) / A < len(male[itm]) / B:

			F2 += len(female[itm])
			female_smaller_group.append(female[itm])
			M2 += len(male[itm])
			male_larger_group.append(male[itm])
			male_d_c += 1
		else:
			male_equal.append(male[itm])
			female_equal.append(female[itm])

		if min(len(female[itm]), len(male[itm])) == 0:
			not_full_cover_count += 1

	# ########################  reweigh dataset ################################

	max_d = min(int(F1 * 1.0 * 0.5), int(M2 * 1.0 * 0.5))
	(male_ans, female_ans) = ([], [])

	(max_answer_cnt, max_best_answer_cnt) = (0, 0)
	for i in range(max_d):
		ans = fsolve(solve_two_groups, [i, i])
		global X, Y
		(X, Y) = (ans[0], ans[1])

				# print('diff1 should close to 0', (F1/A - M1/B)*r - (F1-X)/(A-X)+(M1)/(B-Y))
				# print('diff2 should close to 0', (M2/B - F2/A)*r - (M2-Y)/(B-Y)+(F2)/(A-X))
		# keep looking good reweight results

		if X > 0 and Y > 0 or X == 0 and Y == 0:

			# metrics to measure the quality of the reweight

			(check_answers, cnt) = (0, 0)
			(deletion_diff, duplicate_diff) = (0.0, 0.0)

			# answer to store the data

			(male_tmp_ans, female_tmp_ans) = ([], [])
			for itm in comb_list:
				if len(female[itm]) / A > len(male[itm]) / B:
					global F11, M11
					F11 = len(female[itm])
					M11 = len(male[itm])
					ans2 = fsolve(solve_x, 1.0)
					if ans2[0] >= 0:
						check_answers += 1
						female_tmp_ans.append(round(ans2[0]))
						if ans2[0] > len(female[itm]):
							deletion_diff += abs(ans2[0]
									- len(female[itm]))
						else:
							deletion_diff += abs(ans2[0]
									- round(ans2[0]))
					else:
						female_tmp_ans.append(0)
						duplicate_diff += abs(ans2[0])

					if ans2[0] >= 1:
						cnt += 1
				elif len(female[itm]) / A < len(male[itm]) / B:
					global F22, M22
					F22 = len(female[itm])
					M22 = len(male[itm])
					ans2 = fsolve(solve_y, 1.0)
					if ans2[0] >= 0:
						check_answers += 1
						male_tmp_ans.append(round(ans2[0]))
						if ans2[0] > len(male[itm]):
							deletion_diff += abs(ans2[0]
									- len(male[itm]))
						else:
							deletion_diff += abs(ans2[0]
									- round(ans2[0]))
					else:
						male_tmp_ans.append(0)
						duplicate_diff += abs(ans2[0])

					if ans2[0] >= 1:
						cnt += 1
			if max_answer_cnt < check_answers:
				max_answer_cnt = check_answers
				max_best_answer_cnt = cnt
				male_ans = male_tmp_ans
				female_ans = female_tmp_ans
			elif max_answer_cnt == check_answers \
				and max_best_answer_cnt < cnt:

				max_best_answer_cnt = cnt
				male_ans = male_tmp_ans
				female_ans = female_tmp_ans

	print(('total deletion', sum(male_ans) + sum(female_ans)))
	print(('reweigh quality metric 0 (bigger, better)',
		   'answer > 0 ratio:', max_answer_cnt * 1.0 / (female_d_c
		   + male_d_c) * 1.0, 'answer > 1 ratio', max_best_answer_cnt
		   * 1.0 / (female_d_c + male_d_c) * 1.0))
	print(('reweigh quality metric 1 (smaller, better)',
		   'answer > 0 diff', deletion_diff, 'answer < 0 diff',
		   duplicate_diff))

	reweigh_data = []
	for i in range(len(female_larger_group)):
		total_record = len(female_larger_group[i])
		need_to_delete = int(female_ans[i])
		remain_record = total_record - need_to_delete

		for j in range(remain_record):
			reweigh_data.append(female_larger_group[i][np.random.randint(total_record)][-1])
		for j in range(len(male_smaller_group[i])):
			reweigh_data.append(male_smaller_group[i][j][-1])

	for i in range(len(male_larger_group)):
		total_record = len(male_larger_group[i])
		need_to_delete = int(male_ans[i])
		remain_record = total_record - need_to_delete

		for j in range(remain_record):
			reweigh_data.append(male_larger_group[i][np.random.randint(total_record)][-1])
		for j in range(len(female_smaller_group[i])):
			reweigh_data.append(female_smaller_group[i][j][-1])

	print(('reweigh dataset size', len(reweigh_data)))
	return original_data[reweigh_data, :]


def reverse_reweigh(data, original_data):

	assert args.ratio > 1.0, \
		'reweigh ratio should be larger than 1.0 if its reverse reweigh'

	global F1, M1, F2, M2, A, B, r
	r = args.ratio
	(
		F1,
		M1,
		F2,
		M2,
		A,
		B,
		) = (
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		)

	# ####################### find distribution difference ####################

	comb_list = set()
	for itm in data:
		tmp = ''
		for j in range(len(itm)):
			if j != args.sens_att_index and j != len(data[0]) - 1 and j \
				!= len(data[0]) - 2:
				tmp += str(itm[j]) + '-'
		comb_list.add(tmp)

	(male, female) = ({}, {})
	for itm in comb_list:
		(male[itm], female[itm]) = ([], [])

	for itm in data:
		tmp = ''
		for j in range(len(itm)):
			if j != args.sens_att_index and j != len(data[0]) - 1 and j \
				!= len(data[0]) - 2:
				tmp += str(itm[j]) + '-'
		if itm[args.sens_att_index] == 0:
			B += 1.0
			male[tmp].append(itm)
		else:
			A += 1.0
			female[tmp].append(itm)

	(female_equal, female_smaller_group, female_larger_group) = ([],
			[], [])
	(male_smaller_group, male_equal, male_larger_group) = ([], [], [])
	(not_full_cover_count, female_d_c, male_d_c) = (0, 0, 0)

	for itm in comb_list:
		if len(female[itm]) / A > len(male[itm]) / B:
			F1 += len(female[itm])
			female_larger_group.append(female[itm])
			M1 += len(male[itm])
			male_smaller_group.append(male[itm])
			female_d_c += 1
		elif len(female[itm]) / A < len(male[itm]) / B:

			F2 += len(female[itm])
			female_smaller_group.append(female[itm])
			M2 += len(male[itm])
			male_larger_group.append(male[itm])
			male_d_c += 1
		else:
			male_equal.append(male[itm])
			female_equal.append(female[itm])

		if min(len(female[itm]), len(male[itm])) == 0:
			not_full_cover_count += 1  # if one group has no records, cannot do anything for it

	# ########################  reweigh dataset ################################

	max_d = min(int(F1 * 1.0 * 0.5), int(M2 * 1.0 * 0.5))
	(male_ans, female_ans) = ([], [])
	(max_answer_cnt, max_best_answer_cnt) = (0, 0)

	for i in range(max_d):
		ans = fsolve(solve_two_groups, [i, i])
		global X, Y
		(X, Y) = (ans[0], ans[1])

		# the difference should be close to 0, otherwise the fsolve function is not working properly cause there is no convergence

		if abs((F1 / A - M1 / B) * r - (F1 - X) / (A - X) + M1 / (B
			   - Y)) > 0.000001 or abs((M2 / B - F2 / A) * r - (M2 - Y)
				/ (B - Y) + F2 / (A - X)) > 0.00001:
			continue

		# we don't allow reweigh in large scale

		if abs(X) > args.max_reweigh_size or abs(Y) \
			> args.max_reweigh_size:
			continue

		print(('reweight answer for group X and Y:', X, Y))
		print(('diff1 should close to 0', (F1 / A - M1 / B) * r - (F1
			   - X) / (A - X) + M1 / (B - Y)))
		print(('diff2 should close to 0', (M2 / B - F2 / A) * r - (M2
			   - Y) / (B - Y) + F2 / (A - X)))

		(check_answers, cnt) = (0, 0)

		for itm in comb_list:
			if len(female[itm]) / A > len(male[itm]) / B:
				global F11, M11
				F11 = len(female[itm])
				M11 = len(male[itm])
				ans2 = fsolve(solve_x, 1.0)
				if ans2[0] > 0:
					female_ans.append(0)
				else:
					female_ans.append(round(ans2[0]))
			elif len(female[itm]) / A < len(male[itm]) / B:

				global F22
				global M22
				F22 = len(female[itm])
				M22 = len(male[itm])
				ans2 = fsolve(solve_y, 1.0)
				if ans2[0] > 0:
					male_ans.append(0)
				else:
					male_ans.append(round(ans2[0]))

		# just need one reasonable answer since we are doing duplication

		break

	reweigh_data = []
	for i in range(len(female_larger_group)):
		total_record = len(female_larger_group[i])
		need_to_delete = 0
		if len(female_ans) > 0:
			need_to_delete = int(female_ans[i])
		remain_record = total_record - need_to_delete

		for j in range(remain_record):
			if total_record > 0:
				reweigh_data.append(female_larger_group[i][np.random.randint(total_record)][-1])
		for j in range(len(male_smaller_group[i])):
			reweigh_data.append(male_smaller_group[i][j][-1])

	for i in range(len(male_larger_group)):
		total_record = len(male_larger_group[i])
		need_to_delete = 0
		if len(male_ans) > 0:
			need_to_delete = int(male_ans[i])
		remain_record = total_record - need_to_delete

		for j in range(remain_record):
			if total_record > 0:
				reweigh_data.append(male_larger_group[i][np.random.randint(total_record)][-1])
		for j in range(len(female_smaller_group[i])):
			reweigh_data.append(female_smaller_group[i][j][-1])

	print(('reweigh dataset size', len(reweigh_data)))
	return original_data[reweigh_data, :]


def reweigh_by_class(class_num):

	# decide the number of variables

	original_data = pd.read_csv(args.datasetpath + args.originalfile,
								header=None, delimiter=',')
	original_data = original_data.values

	new_data = []
	data = pd.read_csv(args.datasetpath + args.inputfile, header=None,
					   delimiter=',')
	data = data.values

	for i in range(class_num):

		# data[:,-1] is the index of the data after running the kmeans clustering, so data[:-2] is the label

		subdata = data[data[:, -2] == i]
		if args.reverse:
			rd = reverse_reweigh(subdata, original_data)
		else:
			rd = normal_reweigh(subdata, original_data)
		print('class, reweigh size and attribute number', i, len(rd), \
			len(rd[0]))
		for itm in rd:
			new_data.append(itm)

	new_data = np.array(new_data)
	shuffleIndex = list(range(len(new_data)))
	np.random.shuffle(shuffleIndex)
	new_data = new_data[shuffleIndex]

	f = open(args.datasetpath + args.outputfile, 'w')
	for itm in new_data:
		for j in range(len(itm) - 1):
			f.write('%s,' % itm[j])
		f.write('%s\n' % itm[-1])
	f.close()


def reweigh_all():

	# decide the number of variables

	original_data = pd.read_csv(args.datasetpath + args.originalfile,
								header=None, delimiter=',')
	original_data = original_data.values

	new_data = []
	data = pd.read_csv(args.datasetpath + args.inputfile, header=None,
					   delimiter=',')
	data = data.values

	if args.reverse:
		rd = reverse_reweigh(data, original_data)
	else:
		rd = normal_reweigh(data, original_data)
	print(('all class reweigh, data size, attribute num', len(rd),
		   len(rd[0])))
	for itm in rd:
		new_data.append(itm)

	new_data = np.array(new_data)
	shuffleIndex = list(range(len(new_data)))
	np.random.shuffle(shuffleIndex)
	new_data = new_data[shuffleIndex]

	f = open(args.datasetpath + args.outputfile, 'w')
	for itm in new_data:
		for j in range(len(itm) - 1):
			f.write('%s,' % itm[j])
		f.write('%s\n' % itm[-1])


def one_hot_transfer(all_data, data, cate_feat_index):

	(enc_data, cate_data, num_data) = ([], [], [])
	for itm in data:
		(tmp, tmp2) = ([], [])
		for j in range(len(itm)):
			if j in cate_feat_index:
				tmp.append(itm[j])
			else:
				tmp2.append(itm[j])
		cate_data.append(tmp)
		num_data.append(tmp2)

	base_data = []
	for itm in all_data:
		tmp = []
		for j in range(len(itm)):
			if j in cate_feat_index:
				tmp.append(itm[j])
		base_data.append(tmp)

	enc = OneHotEncoder(categories='auto').fit(np.array(base_data))
	for i in range(len(cate_data)):
		x = list(enc.transform([cate_data[i]]).toarray()[0])
		x.extend(list(num_data[i]))
		enc_data.append(x)

	print(('size of encoded data, # of attributes after encoded',
		   len(enc_data), len(enc_data[0])))
	return enc_data


def kmean_feature_transfer():

	all_data = pd.read_csv(args.datasetpath + args.originalfile,
						   header=None, delimiter=',')
	all_data = all_data.values
	data = all_data

	# remove senstive attribtue and label first

	prot_att = [args.sens_att_index, len(data[0]) - 1]

	# get categority feature index after delete the sensitive attribute

	orignal_feat_index = [int(itm) for itm in args.cate_index.split(' '
						  )]
	cate_feat_index = []
	for itm in orignal_feat_index:
		if itm < args.sens_att_index:
			cate_feat_index.append(itm)
		elif itm > args.sens_att_index:
			cate_feat_index.append(itm - 1)

	data_without_prot_att = []
	for itm in data:
		tmp = []
		for i in range(len(itm)):
			if i not in prot_att:
				tmp.append(itm[i])
		data_without_prot_att.append(tmp)

	if args.reweigh_by_class:
		kmeans_by_class(data_without_prot_att, cate_feat_index,
						prot_att, data)
	else:
		kmeans_all_class(data_without_prot_att, cate_feat_index,
						 prot_att, data)


def kmeans_by_class(
	data_without_prot_att,
	cate_feat_index,
	prot_att,
	data,
	):

	data_without_prot_att_per_label = []
	label_number = args.class_num

	for i in range(label_number):
		data_without_prot_att_per_label.append([])

	for i in range(len(data_without_prot_att)):
		data_without_prot_att_per_label[int(data[i][-1])].append(data_without_prot_att[i])

	enc_data = []
	for i in range(label_number):
		print('label ', i, 'data size ', \
			len(data_without_prot_att_per_label[i]))
		enc_data.append(one_hot_transfer(data_without_prot_att,
						data_without_prot_att_per_label[i],
						cate_feat_index))

	group_num = [int(itm) for itm in args.group_num.split(' ')]
	kmeans = []
	for i in range(label_number):
		kmeans.append(KMeans(n_clusters=group_num[i],
					  random_state=0).fit(enc_data[i]))

	group_data = []
	for i in range(label_number):
		group_data.append({})

	for i in range(label_number):
		for j in range(group_num[i]):
			group_data[i][j] = []

	for i in range(label_number):
		for j in range(len(data_without_prot_att_per_label[i])):
			group_data[i][int(kmeans[i].labels_[j])].append(data_without_prot_att_per_label[i][j])

	# generalize feature for each cluster

	(generalize_feat, generalize_index, centers) = ([], [], [])
	for i in range(label_number):
		generalize_feat.append([])
		generalize_index.append([])
		centers.append([])

	for i in range(label_number):
		for j in range(len(data_without_prot_att_per_label[i][0])):
			generalize_feat[i].append({})
			generalize_index[i].append(0)

	for i in range(label_number):
		for j in range(group_num[i]):
			group = group_data[i][j]
			group = np.array(group)
			kmean_data = []
			for k in range(len(group[0])):
				group_k = group[:, k]
				feat = str(min(group_k)) + '-' + str(max(group_k))

				if feat not in generalize_feat[i][k]:
					generalize_feat[i][k][feat] = generalize_index[i][k]
					generalize_index[i][k] += 1

				kmean_data.append(generalize_feat[i][k][feat])
			centers[i].append(kmean_data)

	for i in range(label_number):
		print('center, data size, attribute size', i, ' ', \
			len(centers[i]), len(centers[i][0]))

	record_cnt = []
	for i in range(label_number):
		record_cnt.append(0)

	# add protected att and label back to the kmean data

	record_index = 0
	transfered_data = []
	for i in range(len(data_without_prot_att)):
		tmp = []
		class_label = int(data[i][-1])

			# get cluster label

		label = \
			int(kmeans[class_label].labels_[record_cnt[class_label]])
		record_cnt[class_label] = record_cnt[class_label] + 1

		for j in range(prot_att[0]):
			tmp.append(centers[class_label][label][j])
		tmp.append(data[i][prot_att[0]])
		for j in range(prot_att[0], len(data_without_prot_att[0])):
			tmp.append(centers[class_label][label][j])
		tmp.append(data[i][-1])
		tmp.append(record_index)
		transfered_data.append(tmp)
		record_index += 1

	f = open(args.datasetpath + args.inputfile, 'w')
	for i in range(len(transfered_data)):
		for j in range(len(transfered_data[i]) - 1):
			f.write('%s,' % transfered_data[i][j])
		f.write('%s\n' % transfered_data[i][len(transfered_data[i])
				- 1])
	f.close()
	print('train_size, attribute size', len(transfered_data), \
		len(transfered_data[0]))


def kmeans_all_class(
	data_without_prot_att,
	cate_feat_index,
	prot_att,
	data,
	):

	enc_data = one_hot_transfer(data_without_prot_att,
								data_without_prot_att, cate_feat_index)
	group_num = [int(itm) for itm in args.group_num.split(' ')]
	kmeans = KMeans(n_clusters=group_num[0],
					random_state=0).fit(enc_data)

	group_data = {}

	for j in range(group_num[0]):
		group_data[j] = []

	for j in range(len(data_without_prot_att)):
		group_data[int(kmeans.labels_[j])].append(data_without_prot_att[j])

	# generalize feature for each cluster

	(generalize_feat, generalize_index, centers) = ([], [], [])

	for j in range(len(data_without_prot_att[0])):
		generalize_feat.append({})
		generalize_index.append(0)

	for j in range(group_num[0]):
		group = group_data[j]
		group = np.array(group)
		kmean_data = []
		for k in range(len(group[0])):
			group_k = group[:, k]
			feat = str(min(group_k)) + '-' + str(max(group_k))

			if feat not in generalize_feat[k]:
				generalize_feat[k][feat] = generalize_index[k]
				generalize_index[k] += 1

			kmean_data.append(generalize_feat[k][feat])
		centers.append(kmean_data)

	print('data size, attribute size', ' ', len(centers), \
		len(centers[0]))

	# add protected att and label back to the kmean data

	(record_index, record_cnt) = (0, 0)
	transfered_data = []
	for i in range(len(data_without_prot_att)):
		tmp = []

			# get cluster label

		label = int(kmeans.labels_[record_cnt])
		record_cnt += 1

		# append attribute before sensitve attribue

		for j in range(prot_att[0]):
			tmp.append(centers[label][j])

		# append sensitive attributes

		tmp.append(data[i][prot_att[0]])

		# append remain attributes

		for j in range(prot_att[0], len(data_without_prot_att[0])):
			tmp.append(centers[label][j])
		tmp.append(data[i][-1])
		tmp.append(record_index)
		transfered_data.append(tmp)
		record_index += 1

	f = open(args.datasetpath + args.inputfile, 'w')
	for i in range(len(transfered_data)):
		for j in range(len(transfered_data[i]) - 1):
			f.write('%s,' % transfered_data[i][j])
		f.write('%s\n' % transfered_data[i][len(transfered_data[i])
				- 1])
	f.close()
	print('train_size, attribute size', len(transfered_data), \
		len(transfered_data[0]))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# reweigh setttings

	parser.add_argument('--ratio', type=float, help='reweigh ratio',
						default=0.8)
	parser.add_argument('--reverse', type=int,
						help='reverse reweigh or not, 1 means reverse reweigh'
						, default=0)
	parser.add_argument('--reweigh_by_class', type=int,
						help='split data by label or not', default=0)
	parser.add_argument('--max_reweigh_size', type=int,
						help='the total size of data after reweigh should be smaller than this value'
						, default=100000)

	# kmeans settings

	parser.add_argument('--class_num', type=int,
						help='# of class for the dataset', default=2)
	parser.add_argument('--sens_att_index', type=int,
						help='the sensitive attribute index', default=8)
	parser.add_argument('--cate_index', type=str,
						help='categorical feature index seperated by space'
						, default='1 3 5 6 7 8 9 13')
	parser.add_argument('--group_num', type=str,
						help='the k settings for kmeans, seperated by space'
						, default='90')

	# io settings

	parser.add_argument('--datasetpath', type=str,
						help='data set directory', default='./dataset/'
						)
	parser.add_argument('--outputfile', type=str,
						help='the reweighed data output',
						default='new_adult_data2-reweighing-race.csv')
	parser.add_argument('--inputfile', type=str,
						help='the reweighed data input',
						default='adult-kmeans-byall.csv')
	parser.add_argument('--originalfile', type=str,
						help='the orginal data',
						default='new_adult_data2-all-for-reweigh.csv')
	args = parser.parse_args()

	kmean_feature_transfer()
	if args.reweigh_by_class:
		reweigh_by_class(args.class_num)
	else:
		reweigh_all()
