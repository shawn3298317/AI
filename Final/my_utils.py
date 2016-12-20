import numpy as np
import math
from sklearn.preprocessing import normalize


## my own library


def get_dict(path):
	print 'Geting dict from '+path
	counter = 0
	word_indices = {}
	indices_word = {}
	for line in open(path,'r').read().splitlines():
		word_indices.update({line:counter})
		indices_word.update({counter:line})
		counter += 1
	print 'Dict len is ',len(word_indices)
	print '=================='
	return word_indices, indices_word

def vectorize_label(path, top_k, word_indices):
	the_lines = open(path).read().lower().splitlines()
	totallines = len(the_lines)
	valid_index = np.zeros(totallines).astype(np.bool_)
	oov = np.zeros(totallines).astype(np.int32)
	y=np.zeros((totallines, top_k),dtype=np.float32)
	for i,lines in enumerate(the_lines):
		for j in lines.split():
			if j in word_indices:
				y[i, word_indices[j] ] =1
				valid_index[i] = 1
			else:
				oov[i] = oov[i]+1 
	y_norm = normalize(y, norm='l1')
	return y, y_norm, valid_index, oov



def getmaxlen(X):
	count = 0
	for coco in X:
		if len(coco) > count:
			count = len(coco)
	return count

def buildlabel(Y,tag_num):
	coco = []
	counter = 0
	for data in Y:
		#print counter
		temp = [0]*tag_num
		length = len(data)
		for tag in data:
			temp[tag-1] = float(1)/float(length)
			#temp[tag-1] = 0
			#temp[tag-1] = 1
		coco.append(temp)
		counter += 1
	coco = np.array(coco)
	return coco
	
def map7eval(preds, dtrain):
	actual = dtrain.get_label()
	predicted = preds.argsort(axis=1)
	predicted = np.fliplr(predicted)[:, :7]
	metric = 0.
	for i in range(7):
		metric += np.sum(actual==predicted[:,i])/(i+1)
	metric /= actual.shape[0]
	return 'MAP@7', metric

def count_MAP_total(pred, dtrain):
	y = dtrain.get_label()
	print y[0]
	print np.shape(y)
	print pred[0]
	print np.shape(pred)
	total_AP = np.zeros(len(y))
	for i,row in enumerate(y):
		if np.sum(y)==0:
			total_AP[i] = 0
		else:
			total_AP[i] = mymap(y[i], pred[i, 0:], 7)
	return 'My map@7', float(np.sum(total_AP))/len(y)

def count_pr_total(y, pred, oov, valid_index):
	total_prec = np.zeros(len(valid_index))
	total_reca = np.zeros(len(valid_index))
	nowline = 0
	for i, ind in enumerate(valid_index):
		if ind == False:
			total_prec[i] = 0
			total_reca[i] = 0
		else:
			total_prec[i], total_reca[i] = multi_pr(y[nowline, 0:], pred[nowline, 0:], oov[i])
			nowline += 1
	return float(np.sum(total_prec))/len(valid_index), float(np.sum(total_reca))/len(valid_index)
	#return (total_prec), (total_reca)

def multi_pr(truth, pred, num_oov):
	true_doc_id = np.nonzero(truth)[0]
	pred[pred>=0.5]=1
	pred[pred<0.5]=0
	my_pred = np.nonzero(pred)[0]
	correct = 0
	one_prec = 0
	one_reca = 0
	for i in my_pred:
		if i in true_doc_id:
			correct += 1
	if not len(my_pred)==0:
		one_prec = float(correct)/(len(my_pred))
	one_reca = float(correct)/(len(true_doc_id)+num_oov)
	#another_accu = float(np.sum(np.equal(pred,truth)))/len(pred)
	return one_prec, one_reca

def mymap(truth, one_pred, cut_off):
	#true_doc_id = np.nonzero(truth)[0]
	true_doc_id = truth
	my_retrieve = one_pred.argsort()[::-1][0:cut_off]
	num_to_retr = 1
	correct = 0
	wrong = 0
	total_precision = []
	hit_times = 0
	for wtf,i in enumerate(my_retrieve):
		if i in true_doc_id:
		#	print('position '+str(wtf)+' hit index '+str(i))
			hit_times += 1
			correct += 1
			num_to_retr -= 1
			now_prec = float(correct)/(correct+wrong)
			total_precision.append(now_prec)
		#	print('Precision is '+str(now_prec))
		else:
			wrong += 1
		if (num_to_retr == 0):
			oop = float(sum(total_precision))/hit_times
			now_prec = float(correct)/(correct+wrong)
			return oop, now_prec
	oop = float(sum(total_precision))/hit_times
	return oop
