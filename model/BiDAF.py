import torch
import torch.nn as nn
import torch.nn.functional as F

from model.word_lv_embed import WordEmbed
from model.char_lv_embed import CharEmbed
from model.highway import Highway
from model.attention_layer import AttnEmbed

class BiDAF(nn.Module):
    def __init__(self, args):
        super(BiDAF, self).__init__()
        self.d = 2 * args.word_embd_dim # word_level + character_level
        self.char_embd_model = CharEmbed(args)
        self.word_embd_model = WordEmbed(args)
        self.highway_model = Highway(self.d)
        self.ctx_embd_model = nn.GRU(self.d, self.d, dropout = 0.2, bidirectional = True, batch_first = True)
        self.attn_embd_model  = AttnEmbed(self.d)
        self.modeling = nn.GRU(8 * self.d, self.d, num_layers = 2, dropout = 0.2, bidirectional = True, batch_first = True)
        self.startIdx = nn.Linear(10 * self.d, 1, bias = False)
        self.endIdx_lstm = nn.GRU(2 * self.d, self.d, dropout = 0.2, bidirectional = True, batch_first = True)
        self.endIdx = nn.Linear(10 * self.d, 1, bias = False)
        
    def _contextualEmbed(self, char_lv, word_lv):
        # batch_first repr. i.e. (batch_size, seq_length, dim)
        char_embding = self.char_embd_model(char_lv)
        word_embding = self.word_embd_model(word_lv)
        concat = torch.cat((char_embding, word_embding), 2)
        highway_embding = self.highway_model(concat)
        ctx_embd, _ = self.ctx_embd_model(highway_embding)
        return ctx_embd
        
    def forward(self, ctx_sent_word, ctx_sent_char, query_word, query_char):
        # 1. Character Embedding Layer
        # 2. Word Embedding Layer
        # 3. Contexual Embedding Layer
        ctx_embding = self._contextualEmbed(ctx_sent_char, ctx_sent_word)
        query_embding = self._contextualEmbed(query_char, query_word)
        
        # 4. Attention Flow layer
        G = self.attn_embd_model(ctx_embding, query_embding)
        
        # 5. Modeling Layer
        M, _ = self.modeling(G)
                      
        # 6. Output Layer
        concat_GM = torch.cat((G, M), 2)
        p1 = F.softmax(self.startIdx(concat_GM).squeeze(), dim = -1)
        
        M2, _ = self.endIdx_lstm(M)
        concat_GM2 = torch.cat((G, M2), 2)
        p2 = F.softmax(self.endIdx(concat_GM2).squeeze(), dim = -1)        
        return p1, p2