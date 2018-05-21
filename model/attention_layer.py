import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnEmbed(nn.Module):
	def __init__(self, input_dim):
		super(AttnEmbed, self).__init__()
		self.scalar_fun = nn.Linear(6 * input_dim, 1, bias = False)

	def forward(self, ctx_embd, query_embd):
		# input shape: (batch_size, seq_len, dim)
		batch_size = ctx_embd.size(0)
		T = ctx_embd.size(1)
		J = query_embd.size(1)
		dim = ctx_embd.size(2)

		# make the shape of similarity matrix: (batch_size, T, J, dim)
		ctx_embd_expand = ctx_embd.unsqueeze(2)
		ctx_embd_expand = ctx_embd_expand.expand(batch_size, T, J, dim)
		query_embd_expand = query_embd.unsqueeze(1)
		query_embd_expand = query_embd_expand.expand(batch_size, T, J, dim)

		# compute similarity matrix
		concat = torch.cat((ctx_embd_expand, query_embd_expand, torch.mul(ctx_embd_expand, query_embd_expand)), 3)
		S = self.scalar_fun(concat).view(batch_size, T, J) # (batch_size, T, J)

		# C2Q
		a = F.softmax(S, dim = -1)
		c2q = torch.bmm(a, query_embd)

		# Q2C
		b = F.softmax(torch.max(S, 2)[0], dim = -1).unsqueeze(1)
		q2c = torch.bmm(b, ctx_embd)
		q2c = q2c.repeat(1, T, 1)

		G = torch.cat((ctx_embd, c2q, ctx_embd.mul(c2q), ctx_embd.mul(q2c)),2)

		return G