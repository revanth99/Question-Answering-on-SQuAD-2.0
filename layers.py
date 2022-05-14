import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.w_embed = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        self.c_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.char_conv = nn.Conv2d(1, 100, (100, 5))
        self.hwy = HighwayEncoder(2, hidden_size * 2)

    def forward(self, x, y):
        batch_size = x.size(0)

        w_emb = self.w_embed(x)  
        w_emb = F.dropout(w_emb, self.drop_prob, self.training)
        w_emb = self.proj(w_emb)

        c_emb = self.c_embed(y)
        c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        c_emb = c_emb.view(-1, 100, c_emb.size(2)).unsqueeze(1)
        c_emb = self.char_conv(c_emb).squeeze()
        c_emb = F.max_pool1d(c_emb, c_emb.size(2)).squeeze()
        c_emb = c_emb.view(batch_size, -1, 100)

        emb = torch.cat([w_emb, c_emb], dim=-1)

        emb = self.hwy(emb)
        return emb


class HighwayEnc(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(HighwayEnc, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEnc(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.9):
        super(RNNEnc, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        orig_len = x.size(1)

        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     
        x = pack_padded_sequence(x, lengths, batch_first=True)


        x, _ = self.rnn(x)  
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  

        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def masked_softmax(logits, mask, dim=-1, log_softmax=False):
        mask = mask.type(torch.float32)
        masked_logits = mask * logits + (1 - mask) * -1e30
        softmax_fn = F.log_softmax if log_softmax else F.softmax
        probs = softmax_fn(masked_logits, dim)

        return probs

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        
        c_mask = c_mask.view(batch_size, c_len, 1)  
        q_mask = q_mask.view(batch_size, 1, q_len)  
        s1 = masked_softmax(s, q_mask, dim=2)       
        s2 = masked_softmax(s, c_mask, dim=1)       

        
        a = torch.bmm(s1, q)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  

        return x

    def get_similarity_matrix(self, c, q):
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  
        q = F.dropout(q, self.drop_prob, self.training)  

        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
        
    


class BiDAFOutput(nn.Module):
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
