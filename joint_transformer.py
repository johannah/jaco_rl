# started from working example of gpt2 -
# https://github.com/graykode/gpt-2-Pytorch
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pickle
import numpy as np
from IPython import embed
from torch.nn.parameter import Parameter
import copy
import math

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    def __init__(self, n_state, n_ctx, n_head, scale=False):
        super(Attention, self).__init__()
        #state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, n_state)
        self.c_proj = Conv1D(n_state, n_state)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, n_embed):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        self.c_fc = Conv1D(n_state, n_embed)
        self.c_proj = Conv1D(n_embed, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):

    def __init__(self, n_ctx, n_embed, layer_norm_epsilon, n_head, scale=False):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(n_embed, eps=layer_norm_epsilon)
        self.attn = Attention(n_embed, n_ctx, n_head, scale)
        self.ln_2 = LayerNorm(n_embed, eps=layer_norm_epsilon)
        self.mlp = MLP(4 * n_embed, n_embed)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, n_vocab, n_timesteps, n_embed, n_layer=12, layer_norm_epsilon=1e-5, n_ctx=1024, n_head=12):
        super(GPT2Model, self).__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Embedding(n_timesteps, n_embed)
        block = Block(n_ctx, n_embed, layer_norm_epsilon, n_head, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embed, eps=layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.token_embedding(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, n_embed):
        super(GPT2LMHead, self).__init__()
        self.n_embed = n_embed
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embed)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2LMHeadModel(nn.Module):
    def __init__(self, n_vocab, n_timesteps, n_embed, n_layer=12, layer_norm_epsilon=1e-5, n_ctx=1024, n_head=12):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(n_vocab=n_vocab, n_timesteps=n_timesteps, n_embed=n_embed, n_layer=n_layer, layer_norm_epsilon=layer_norm_epsilon, n_ctx=n_ctx, n_head=n_head)
        self.lm_head = GPT2LMHead(self.transformer.token_embedding.weight, n_embed)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.token_embedding.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents

#def train():
#    model.train() # Turn on the train mode
#    total_loss = 0.
#    start_time = time.time()
#    ntokens = len(TEXT.vocab.stoi)
#    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
#        # data shape is 35, 28
#        # target shape is 980
#        data, targets = get_batch(train_data, i)
#        optimizer.zero_grad()
#        # ntokens is 28785
#        # output shape is [35,28,28785]
#        output = model(data)
#        # flattens out output output.view(-1, ntokens) is [980,28785]
#        loss = criterion(output.view(-1, ntokens), targets)
#        loss.backward()
#        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#        optimizer.step()
#
#        total_loss += loss.item()
#        log_interval = 200
#        if batch % log_interval == 0 and batch > 0:
#            cur_loss = total_loss / log_interval
#            elapsed = time.time() - start_time
#            print('| epoch {:3d} | {:5d}/{:5d} batches | '
#                  'lr {:02.2f} | ms/batch {:5.2f} | '
#                  'loss {:5.2f} | ppl {:8.2f}'.format(
#                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
#                    elapsed * 1000 / log_interval,
#                    cur_loss, math.exp(cur_loss)))
#            total_loss = 0
#            start_time = time.time()
#
#def evaluate(eval_model, data_source):
#    eval_model.eval() # Turn on the evaluation mode
#    total_loss = 0.
#    ntokens = len(TEXT.vocab.stoi)
#    with torch.no_grad():
#        for i in range(0, data_source.size(0) - 1, bptt):
#            data, targets = get_batch(data_source, i)
#            output = eval_model(data)
#            output_flat = output.view(-1, ntokens)
#            total_loss += len(data) * criterion(output_flat, targets).item()
#    return total_loss / (len(data_source) - 1)
#
def batchify(filepath='results/random_jaco_00000_0000149999_buffer.pkl', batch_size=32):

    # ....code for testing reshape
    #aa = np.array([np.arange(6) for a in range(50000)])
    #for xx in range(aa.shape[0]):
    #    if not xx%500:
    #        aa[xx]*=10
    #aa[1500]*=100
    #now data is shape 500,100,6
    #raa = aa.ravel().reshape(100,500,6).swapaxes(0,1)
    #now data is shape 3000,100
    #rab = aa.ravel().reshape(100,3000).swapaxes(0,1)
    # should be :
    # In [69]: rab[:10,3]
    # Out[69]: array([   0, 1000, 2000, 3000, 4000, 5000,    0,    1,    2,    3])

    # in dm_control dataset - episodes are exactly 500 timesteps (50hz*10s max
    # time step) - this will change and should not be hardcoded
    replay = pickle.load(open(filepath, 'rb'))
    max_ind = max([replay.ptr, replay.size])
    # make joints sequential
    n_episodes = max_ind//ep_len
    flat_ep_len = int(ep_len*n_joints)
    # we only are using the first n_joints
    # arm_pos is 13,2 in shape, flattens to first 26
    # arm_vel is 13
    pos_mat = np.arange(n_pos_bins*n_pos_bins, dtype=np.int32).reshape(n_pos_bins, n_pos_bins)
    arm_pos_states = replay.state[:max_ind,:n_joints*2]
    darm_pos_states = (((arm_pos_states+1)/2.0)*n_pos_bins).astype(np.int32)

    embed()
    #arm_vel_states = replay.state[:max_ind,26:26+n_joints]
    #darm_vel_states = (((arm_vel_states+1)/2.0)*n_vel_bins).astype(np.int32)
    # build dataset in the form of:
    # sequence, bs, features
    dactions = (((replay.action[:max_ind,:n_joints]+1)/2.0)*n_action_bins).astype(np.int32)
    e0 = dactions[0]
    e1 = dactions[ep_len*1]
    e2 = dactions[ep_len*2]
    ep_actions = dactions.ravel().reshape(n_episodes,ep_len,n_joints).swapaxes(0,1)
    ep_vel_states = darm_vel_states.ravel().reshape(n_episodes,ep_len,n_joints).swapaxes(0,1)
    ep_pos_states = darm_pos_states.ravel().reshape(n_episodes,ep_len,n_joints*2).swapaxes(0,1)
    # test our dataset logic
    assert (ep_actions[0,0] == e0).sum() == n_joints
    assert (ep_actions[0,1] == e1).sum() == n_joints
    assert (ep_actions[0,2] == e2).sum() == n_joints


if __name__ == '__main__':
    device = 'cpu'
    # chunks are along dim 0
    ep_len = 500
    n_joints = 6
    seq_len = ep_len*n_joints
    bptt = 35
    n_action_bins = 40
    n_pos_bins = 100
    n_vel_bins = 40
    ## after batchify, the data looks like [seq, batch_size]
    batchify('results/random_test/random_jaco_193488_0000049999_buffer.pkl')

    model = GPT2LMHeadModel(n_pos_bins=n_pos_bins, n_vel_bins=n_vel_bins,
                            n_action_bins=n_action_bins, n_timesteps=seq_len, n_embed=768,
                            n_layer=12, layer_norm_epsilon=1e-5, n_ctx=1024,
                            n_head=12).to(device)
    lr = 5.0 # learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    epochs = 300 # The number of epochs
    best_model = None

    batch_size = 28
    eval_batch_size = 10



    #for epoch in range(1, epochs + 1):
    #    epoch_start_time = time.time()
    #    train()
    #    val_loss = evaluate(model, val_data)
    #    print('-' * 89)
    #    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
    #          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
    #                                     val_loss, math.exp(val_loss)))
    #    print('-' * 89)

    #    if val_loss < best_val_loss:
    #        best_val_loss = val_loss
    #        best_model = model

    #    scheduler.step()
