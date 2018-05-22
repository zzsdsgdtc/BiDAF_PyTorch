import os
import shutil
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from util.process_data import load_processed_json, load_glove_weights, DataSet
from model.BiDAF import BiDAF
from util.ema import EMA
from train import Trainer

# cmd parser
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=12, help='num of epoch to run')
parser.add_argument('--batch_size', type=int, default=60, help='input batch size')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')
parser.add_argument('--word_embd_dim', type=int, default=100, help='word embedding size')
parser.add_argument('--char_embd_dim', type=int, default=8, help='character embedding size')
parser.add_argument('--start_epoch', type=int, default=0, help='resume epoch count, default=0')
parser.add_argument('--test', type=bool, default=False, help='True to enter test mode')
parser.add_argument('--resume', default='~/checkpoints/Epoch-11.model', type=str, metavar='PATH', help='path of saved params')
args = parser.parse_args()

# loading data
print("loading data.....\n\n")
home = os.path.expanduser("~")
train_json, train_shared_json = load_processed_json('./data/squad/data_train.json', './data/squad/shared_train.json')
test_json, test_shared_json = load_processed_json('./data/squad/data_test.json', './data/squad/shared_test.json')
train_data = DataSet(train_json, train_shared_json)
test_data = DataSet(test_json, test_shared_json)

# make *_to_index combining both training and test set
w2i_train, c2i_train = train_data.get_word_index()
w2i_test, c2i_test = test_data.get_word_index()
word_vocab = sorted(list(set(list(w2i_train.keys()) + list(w2i_test.keys()))))
w2i = {w : i for i, w in enumerate(word_vocab, 3)} # 0:NULL, 1: UNK, 2: ENT
char_vocab = sorted(list(set(list(c2i_train.keys()) + list(c2i_test.keys()))))
c2i = {c : i for i, c in enumerate(char_vocab, 3)}
NULL = "-NULL-"
UNK = "-UNK-"
ENT = "-ENT-"
w2i[NULL] = 0
w2i[UNK] = 1
w2i[ENT] = 2
c2i[NULL] = 0
c2i[UNK] = 1
c2i[ENT] = 2

# load pre-trained GloVe
glove_path = os.path.join(home, "data", "glove")
glove = torch.from_numpy(load_glove_weights(glove_path, args.word_embd_dim, len(w2i), w2i)).type(torch.FloatTensor)

# set up arguments
args.word_vocab_size = len(w2i)
args.char_vocab_size = len(c2i)
args.pretrained = True
args.pretrained_embd = glove
## for CNN
args.filters = [[1, 5]]
args.out_chs = 100

def main(args):
	model = BiDAF(args)
	if torch.cuda.is_available():
		model.cuda()

	optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)

	# check if resume
	if os.path.isfile(args.resume):
		print(">>>>>>>>>>>>> loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print(">>>>>>>>>>>>> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
	else:
		print(">>>>>>>>>>>>> no checkpoint found at '{}'".format(args.resume))

	# exponential moving average
	ema = EMA(0.999)
	for name, parameter in model.named_parameters():
		if parameter.requires_grad:
			ema.register(name, parameter.data)

	# debugging info
	# print('parameter size:')
	# for name, param in model.named_parameters():
	# if param.requires_grad:
	# 	print(name, param.data.size())

	if args.test:
		print(">>>>>>>>>>>>> Test mode")
		pass  # TODO add test module
	else:
		print(">>>>>>>>>>>>> Train mode")
		# (model, data, w2i, c2i, optimizer, ema, epoch, starting_epoch = 0, batch_size)
		trainer = Trainer(model, train_data, w2i, c2i, optimizer, ema, args.epoch, args.start_epoch, args.batch_size)
		trainer.train()

print("starting main()........\n\n")
main(args)
# if __name__ == "__main__":
#     main(args)
