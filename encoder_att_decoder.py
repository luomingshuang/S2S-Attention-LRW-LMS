import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

from torch.distributions import Categorical
from data import MyDataset

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):

        outputs, hidden = self.gru(src, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn1 = nn.Linear(self.hidden_size * 2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1) 
    
    def _init_hidden(self):
        nn.init.xavier_normal_(self.attn1.weight)
        nn.init.xavier_normal_(self.attn2.weight)

    def forward(self, hidden, encoder_outputs): 
        seq_len, batch_size, _ = encoder_outputs.size()
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        inputs = torch.cat((encoder_outputs, h), 2).view(-1, self.hidden_size*2)
        o = self.attn2(F.tanh(self.attn1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)
        return context

    # def score(self, hidden, encoder_outputs):
    #     # [B*T*2H]->[B*T*H]
    #     assert(hidden.size(2) + encoder_outputs.size(2) == self.hidden_size*2)
    #     energy = self.attn(torch.cat([hidden, encoder_outputs], 2)).softmax(-1)
    #     energy = energy.transpose(1, 2)  # [B*H*T]
    #     v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
    #     #print('')
    #     energy = torch.bmm(v, energy)  # [B*1*T]
    #     return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        # print('last_hidden size is:', last_hidden.size())>>>[3, 16, (1024)hidden_size]
        # print('last_hidden[-1] size is:', last_hidden[-1].size())>>>[1, 16, 1024]
        context = self.attention(last_hidden[-1], encoder_outputs)
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        # context = context.transpose(0, 1)  # (1,B,N)
        #print('context size is:', context.size())
        # Combine embedded input word and attended context, run through RNN
        #print('embedded size is:', embedded.size())
        context = context.transpose(0, 1)
        rnn_input = torch.cat([embedded, context], 2)
        #print('rnn_input size is:', rnn_input.size())
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden

import numpy as np
eps = np.finfo(np.float32).eps.item()
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        if(not trg is None):
            max_len = trg.size(0)
        else:
            max_len = src.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        # print('encoder_output size is:', encoder_output.size())
        # print('hidden size is:', hidden.size())

        hidden = hidden[:self.decoder.n_layers]
        #print('hidden hou size is:', hidden.size())
        output = torch.zeros(src.size(1)).long().cuda()  # sos

        RL_logprobs = Variable(torch.zeros(max_len, batch_size)).cuda()
        RL_preds = Variable(torch.zeros(max_len, batch_size)).cuda()
        Greedy_preds = Variable(torch.zeros(max_len, batch_size)).cuda()
        ED_rl = Variable(torch.zeros(max_len, batch_size)).cuda()
        reward_rl = Variable(torch.zeros(max_len, batch_size)).cuda()
        ED_greedy = Variable(torch.zeros(max_len, batch_size)).cuda()
        reward_greedy = Variable(torch.zeros(max_len, batch_size)).cuda()
        reward_rl_expected = Variable(torch.zeros(max_len, batch_size)).cuda()
        reward_greedy_expected = Variable(torch.zeros(max_len, batch_size)).cuda()

        SOS = Variable(torch.ones(batch_size)).cuda()

        tru_length = Variable(torch.zeros(batch_size)).cuda()

        reward = Variable(torch.zeros(max_len, batch_size)).cuda()
        #get the length of the truth text
        if (not trg is None):
            trg_pred = trg.transpose(0,1).contiguous()
            tru_text = MyDataset.arr2txt(trg_pred)
            for t in range(batch_size):
                tru_length[t] = len(tru_text[t])

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_output)
            #print('decoder hidden size is:', hidden.size())
            outputs[t] = output
            if(not trg is None):
                is_teacher = random.random() < teacher_forcing_ratio
            else:
                is_teacher = False
            #greedy decoder
            # greedy_top1 = output.data.max(1)[1]
            # Greedy_preds[t] = greedy_top1
            # Greedy_predict = Greedy_preds.transpose(0,1).contiguous()

            #reinforce decoder
            # print('output is:', output)
            
            ########下面是有采样的程序########################################
            # rl_output = torch.exp(output)
            # m = Categorical(rl_output)
            # rl_top1 = m.sample()
            # RL_preds[t] = rl_top1
            # RL_logprobs[t] = m.log_prob(rl_top1)
            # RL_preds[0] = SOS
            # RL_predict = RL_preds.transpose(0,1).contiguous()
            ###############################################################

            #########下面是不采样的程序###############################################
            rl_output = torch.exp(output)
            rl_top1 = rl_output.data.max(1)[1]
            m = Categorical(rl_output)
            RL_preds[t] = rl_top1
            RL_logprobs[t] =  m.log_prob(rl_top1)
            RL_preds[0] = SOS
            RL_predict = RL_preds.transpose(0,1).contiguous()
            #######################################################################

            if (not trg is None):
                # for j in range(batch_size):
                #     # print(rl_top1, trg[t])
                #     if rl_top1[j] == trg[t][j]:
                #         reward_rl[t,j] = torch.tensor(1).cuda()
                #     else:
                #         reward_rl[t,j] = torch.tensor(0).cuda()            
                #############################
                trg_preds = trg.transpose(0,1).contiguous()
                # print('trg_preds is:', trg_preds, trg_preds.size())
                RL_text = MyDataset.tensor2text(RL_predict)
                # print('RL_text is:', RL_text)
                # Greedy_text = MyDataset.arr2txt(Greedy_predict)
                tru_text = MyDataset.arr2txt(trg_preds)
                # print('tru_text is:', tru_text)
                ED_rl[t] = torch.tensor(MyDataset.ED(RL_text, tru_text))
                # print('ED_rl[t] is:', ED_rl[t])
                ED_rl[t] = ED_rl[t] 
                if t == 1:
                    reward_rl[t] = -(ED_rl[t] - tru_length)
                else:
                    reward_rl[t] = -(ED_rl[t] - ED_rl[t-1])
                ##############################
                        
            output = Variable(rl_top1).cuda()
            # output = Variable(trg.data[t] if is_teacher else top1).cuda()
        
        # print('RL_preds is:', RL_preds)
        # set 0 or 1 as the reward
        # print('reward_rl is:', reward_rl, reward_rl.size())
        R1 = Variable(torch.zeros(batch_size)).cuda()
        reward_mean = Variable(torch.zeros(batch_size)).cuda()
        reward_std = Variable(torch.zeros(batch_size)).cuda()
        gamma = 0.95
        for t in reversed(range(max_len)):
            if t == 0:
                reward_rl_expected[t] = Variable(torch.zeros(batch_size)).cuda()
            else:
                r = reward_rl[t]
                R1 = r + gamma * R1
                reward_rl_expected[t] = R1
        reward_rl_expected = reward_rl_expected / tru_length        
        reward_rl_expected_mean = reward_rl_expected.transpose(0,1).contiguous()
        
        for t in range(batch_size):
            # print('reward_rl_expected_mean is:', reward_rl_expected_mean[t], reward_rl_expected_mean[t].sum())
            reward_mean[t] = (reward_rl_expected_mean[t][1:].mean())
            reward_std[t] = (reward_rl_expected_mean[t][1:].std())
        # print('reward_rl_expected is:', reward_rl_expected)
        for t in range(1, max_len):
            reward_rl_expected[t] = (reward_rl_expected[t] - reward_mean) / (reward_std + eps) 
        # print('RL_logprobs is:', RL_logprobs)
        # print('reward_rl_expected is:', reward_rl_expected)
        policy_loss = []
        # print(RL_logprobs.shape, reward_rl_expected.shape)
        for log_prob, reward in zip(RL_logprobs, reward_rl_expected):
            policy_loss.append(-log_prob * reward)
        rl_loss = torch.cat(policy_loss).sum()
        # print('rl_loss is:', rl_loss)
        # print('reward_mean is:', reward_mean)
        # print('reward_std is:', reward_std)
        # print('reward_mean is:', reward_mean)
        # print('ED_rl is:', ED_rl, ED_rl.size())
        # print('reward_rl is:', reward_rl, reward_rl.size())    
        # print('RL_logprobs is:', RL_logprobs, RL_logprobs.size())
        
        return outputs, rl_loss

