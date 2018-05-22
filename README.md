# BiDAF_PyTorch
Reproduction of BiDAF using PyTorch

# 1. Pre-processing
## 1.1 Download data and glove
a) chmod +x download.h

b) ./download.h

## 1.2 pre-processing
python -m squad.prepro

# 2. Train
python main.py --batch_size 30

Note that the suggested batch_size is 60 in paper, however 60 will cause cuda out of memory

# 3. Other arguments supported for now
parser.add_argument('--epoch', type=int, default=12, help='num of epoch to run')

parser.add_argument('--batch_size', type=int, default=60, help='input batch size')

parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')

parser.add_argument('--word_embd_dim', type=int, default=100, help='word embedding size')

parser.add_argument('--char_embd_dim', type=int, default=8, help='character embedding size')

parser.add_argument('--start_epoch', type=int, default=0, help='resume epoch count, default=0')

parser.add_argument('--test', type=bool, default=False, help='True to enter test mode')

parser.add_argument('--resume', default='~/checkpoints/Epoch-11.model', type=str, metavar='PATH', help='path of saved params')

# 4. Results for now (after 12 epoch) (only in training set for now)
starting index EM: 65.311%

end index EM: 25.431%

This EM doesn't represent the resulting EM in the paper. And from results there must be bugs in codes
