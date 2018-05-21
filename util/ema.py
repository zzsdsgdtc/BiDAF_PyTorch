import torch.nn as nn

class EMA(nn.Module):
	def __init__(self, decay):
		super(EMA, self).__init__()
		self.decay = decay
		self.shadow = {}

	def register(self, name, parameter):
		self.shadow[name] = parameter.clone()

	def forward(self, name, parameter):
		assert name in self.shadow
		avg = (1 - self.decay) * parameter + self.decay * self.shadow[name]
		self.shadow[name] = avg.clone()
		return avg