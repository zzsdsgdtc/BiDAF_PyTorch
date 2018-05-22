import torch
import torch.nn as nn
import torch.optim as optim
import os.path
import numpy as np
import random
from tqdm import tqdm
import datetime
from util.process_data import get_idx_tensor, DataSet

class Trainer(object):
	def __init__(self, model, data, w2i, c2i, optimizer, ema, epoch, starting_epoch, batch_size):
		self.model = model
		self.data = data
		self.optimizer = optimizer
		self.ema = ema
		self.num_epoch = epoch
		self.start_from = starting_epoch
		self.word_to_index = w2i
		self.char_to_index = c2i
		self.batch_size = batch_size

	def train(self):
		self.model.train()
		for epoch in tqdm(range(self.start_from, self.num_epoch)):
			print(">>>>>>>>>>>>>Processing epoch:", epoch)
			batches = self.data.get_batches(self.batch_size, shuffle = True)
			p1_EM, p2_EM = 0, 0
			num_data_processed = 0
			for i, batch in enumerate(batches):
				# each batch consists of tuples of (ctx_word_lv, ctx_char_lv, query_word_lv, query_char_lv, answer)
				max_ctx_sent_len = max([len(tupl[0]) for tupl in batch])
				max_ctx_word_len = max([len(word) for tupl in batch for word in tupl[1]])
				max_query_sent_len = max([len(tupl[2]) for tupl in batch])
				max_query_word_len = max([len(word) for tupl in batch for word in tupl[3]])

				# padding to make batch equal lengthy
				ctx_word_lv, ctx_char_lv, query_word_lv, query_char_lv, answer = get_idx_tensor(batch, 
																								self.word_to_index, 
																								self.char_to_index, 
																								max_ctx_sent_len, 
																								max_ctx_word_len,
																								max_query_sent_len,
																								max_query_word_len)
				# forward
				ans_start = answer[:, 0]
				ans_end = answer[:, 1] - 1
				p1, p2 = self.model(ctx_word_lv, ctx_char_lv, query_word_lv, query_char_lv)
				loss_p1 = nn.NLLLoss(p1, ans_start)
				loss_p2 = nn. NLLLoss(p2, ans_end)
				loss = torch.add(loss_p1, loss_p2)
				p1_EM += torch.sum(ans_start == torch.max(p1, 1)[1]).item()
				p2_EM += torch.sum(ans_start == torch.max(p2, 1)[1]).item()
				num_data_processed += len(batch)

				# print training process
				if (i + 1) % 50 == 0:
					loss_info = "[{}] Epoch {} completed {:.1f}%, loss_p1: {:.3f}, loss_p2: {:.3f}"
					print(loss_info.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
										   epoch, 100 * i / len(batches),
										   loss_p1.data[0], loss_p2.data[0]))

					EM_info = "p1 EM: {:.3f}% ({}/{}), p2 EM: {:.3f}% ({}/{})"
					print(EM_info.format(100 * p1_EM / num_data_processed, p1_EM, num_data_processed,
										 100 * p2_EM / num_data_processed, p2_EM, num_data_processed))

				# backward
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				for name, parameter in model.named_parameters():
					if parameter.requires_grad:
						parameter.data = ema(name, parameter,data)

			# end of one epoch
			print(">>>>>>>>>>>>>Epoch", epoch, "result")
			print('p1 EM: {:.3f}, p2 EM: {:.3f}'.format(100 * p1_EM / num_data_processed,
													    100 * p2_EM / num_data_processed))
			filename = '{}/Epoch-{}.model'.format('~/checkpoints', epoch)
			torch.save({'epoch': epoch + 1, 
						'state_dict': model.state_dict(), 
						'optimizer': optimizer.state_dict(),
						'p1_EM': 100 * p1_EM / num_data_processed,
						'p2_EM': 100 * p2_EM / num_data_processed
						}, filename=filename)


