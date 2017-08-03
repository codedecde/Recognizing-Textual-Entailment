import Lang as L
from rte_model import RTE
from utils import *
import torch
from sklearn.metrics import accuracy_score
import pdb
import sys
import argparse

# ROOT_DIR = '/home/bass/DataDir/RTE/'
ROOT_DIR = ""

def get_arguments():
	def check_boolean(args, attr_name):
		assert hasattr(args, attr_name), "%s not found in parser"%(attr_name)
		bool_set = set(["true", "false"])
		args_value = getattr(args, attr_name)
		args_value = args_value.lower()
		assert args_value in bool_set, "Boolean argument required for attribute %s"%(attr_name)
		args_value = False if args_value == "false" else True
		setattr(args, attr_name, args_value)
		return args
	
	parser = argparse.ArgumentParser(description='Recognizing Textual Entailment')
	parser.add_argument('-n_embed', action="store", default=300, dest="embedding_dim", type=int)
	parser.add_argument('-n_dim', action="store", default=300, dest="hidden_dim", type=int)
	parser.add_argument('-batch', action="store", default=256, dest="batch_size", type=int)
	parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
	parser.add_argument('-l2', action="store", default=0.0003, dest="l2", type=float)
	parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
	# Using strings as a proxy for boolean flags. Checks happen later
	parser.add_argument('-last_nonlinear', action="store", default = "false", dest="last_nonlinear", type=str)
	parser.add_argument('-train_flag', action="store", default = "true", dest="train_flag", type=str)
	parser.add_argument('-continue_training', action="store", default = "false", dest="continue_training", type=str)	
	parser.add_argument('-wbw_attn', action="store", default = "false", dest="wbw_attn", type=str)

	args = parser.parse_args(sys.argv[1:])
	# Checks for the boolean flags
	args = check_boolean(args, 'last_nonlinear')
	args = check_boolean(args, 'train_flag')
	args = check_boolean(args, 'continue_training')
	args = check_boolean(args, 'wbw_attn')

	return args

def get_options(args):
	options = {}
	# MISC
	options['CLASSES_2_IX'] = {'neutral':1, 'contradiction':2, 'entailment' : 0}
	options['VOCAB'] = ROOT_DIR + 'data/vocab.pkl'
	# options['TRAIN_FILE'] = ROOT_DIR + 'data/train.txt'
	# options['VAL_FILE'] = ROOT_DIR + 'data/dev.txt'
	# options['TEST_FILE'] = ROOT_DIR + 'data/test.txt'

	options['TRAIN_FILE'] = ROOT_DIR + 'data/tinyTrain.txt'
	options['VAL_FILE'] = ROOT_DIR + 'data/tinyVal.txt'
	options['TEST_FILE'] = ROOT_DIR + 'data/tinyVal.txt'

	# Network Properties
	options['LAST_NON_LINEAR'] = args.last_nonlinear if hasattr(args, 'last_nonlinear') else False
	options['EMBEDDING_DIM']   = args.embedding_dim  if hasattr(args, 'embedding_dim')  else 300
	options['HIDDEN_DIM']      = args.hidden_dim     if hasattr(args, 'hidden_dim')     else 300
	options['BATCH_SIZE']      = args.batch_size     if hasattr(args, 'batch_size')     else 256
	options['DROPOUT']         = args.dropout        if hasattr(args, 'dropout')        else 0.1
	options['L2']              = args.l2             if hasattr(args, 'l2')             else 0.0003
	options['LR']              = args.lr             if hasattr(args, 'lr')             else 0.001
	options['WBW_ATTN']        = args.wbw_attn       if hasattr(args, 'wbw_attn')       else False
	# Build the save string 
	if options['WBW_ATTN']:
		options['SAVE_PREFIX'] = ROOT_DIR + 'models_wbw/model'
	else:
		options['SAVE_PREFIX'] = ROOT_DIR + 'models/model'
	options['SAVE_PREFIX'] += '_EMBEDDING_DIM_%d'%(options['EMBEDDING_DIM'])
	options['SAVE_PREFIX'] += '_HIDDEN_DIM_%d'%(options['HIDDEN_DIM'])
	options['SAVE_PREFIX'] += '_DROPOUT_%.4f'%(options['DROPOUT'])
	options['SAVE_PREFIX'] += '_L2_%.4f'%(options['L2'])
	options['SAVE_PREFIX'] += '_LR_%.4f'%(options['LR'])
	options['SAVE_PREFIX'] += '_LAST_NON_LINEAR_%s'%(str(options['LAST_NON_LINEAR']))	
	
	options['TRAIN_FLAG']   = args.train_flag     if hasattr(args, 'train_flag')     else True
	options['CONTINUE_TRAINING'] = args.continue_training if hasattr(args, 'continue_training') else True
	
	return options

args = get_arguments()
options = get_options(args)

l_en = L.Lang('en')
l_en.load_file(options['VOCAB'])

def data_generator(filename, l_en):
	X = []
	y = []
	valid_labels = set(['neutral','contradiction','entailment'])
	unknown_count = 0
	with open(filename) as f:
		for line in f:
			line = line.strip().split('\t')
			if line[2] == '-':
				unknown_count += 1
				continue
			assert line[2] in valid_labels, "Unknown label %s"%(line[2])
			X.append((l_en.tokenize_sent(line[0]), l_en.tokenize_sent(line[1])))
			y.append(line[2])
	print 'Num Unknowns : %d'%(unknown_count)
	return X,y


rte_model = RTE(l_en, options)

if options['TRAIN_FLAG']:	
	print "MODEL PROPERTIES:\n\tEMBEDDING_DIM : %d\n\tHIDDEN_DIM : %d"%(options['EMBEDDING_DIM'], options['HIDDEN_DIM'])
	print "\tDROPOUT : %.4f\n\tL2 : %.4f\n\tLR : %.4f\n\tLAST_NON_LINEAR : %s"%(options['DROPOUT'], options['L2'], options['LR'], str(options['LAST_NON_LINEAR']))
	print "\tWBW ATTN : %s"%(str(options['WBW_ATTN']))
	print 'LOADING DATA ...'
	X_train,y_train = data_generator(options['TRAIN_FILE'], l_en)
	X_val, y_val = data_generator(options['VAL_FILE'], l_en)
	print 'DATA LOADED:\nTRAINING SIZE : %d\nVALIDATION SIZE : %d'%(len(X_train), len(X_val))
	if options['CONTINUE_TRAINING']:
		best_model_file = get_best_model_file(options['SAVE_PREFIX'], model_suffix='.model')
		best_model_state = torch.load(best_model_file)
		rte_model.load_state_dict(best_model_state)
	rte_model.fit(X_train,y_train, X_val, y_val, n_epochs = 5)
else:	
	best_model_file = get_best_model_file(options['SAVE_PREFIX'],model_suffix='.model')
	best_model_state = torch.load(best_model_file)
	
	rte_model.load_state_dict(best_model_state)
	X_test, y_test = data_generator(options['TEST_FILE'], l_en)
	preds_test = rte_model.predict(X_test, options['BATCH_SIZE'], probs = False)
	test_acc = accuracy_score([options['CLASSES_2_IX'][w] for w in y_test], preds_test)
	print "TEST ACCURACY FROM BEST MODEL : %.4f" %(test_acc)
