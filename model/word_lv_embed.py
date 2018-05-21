import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WordEmbed(nn.Module):
	def __init__(self, args):
		super(WordEmbed, self).__init__()
		self.embedding = nn.Embedding(args.word_vocab_size, args.word_embd_dim)
		if args.pretrained:
			self.embedding.weight = nn.Parameter(args.pretrained_embd, requires_grad = False)

	def forward(self, word):
		return self.embedding(word)