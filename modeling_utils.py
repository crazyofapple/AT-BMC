#coding=utf-8

import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from evaluator import EvalElement

def biaffine_mapping(vector_set_1,
                     vector_set_2,
                     num_labels,
                     trans_matrix,
                     add_bias_1=True,
                     add_bias_2=True):
  '''
    Args:
      vector_set_1: [batch_size, seq_length, emb_size]
      vector_set_2: [batch_size, seq_length, emb_size]
      trans_matrix: [emb_size, num_labels, emb_size]
    Returns:
      Logits
  '''
  batch_size = vector_set_1.size(0)
  seq_length = vector_set_1.size(1)
  emb_size = vector_set_1.size(2)
  # num_labels = trans_matrix.size(1)
  
  vector_set_1 = vector_set_1.view((-1, emb_size))  # [batch_size*seq_length, emb_size]
  trans_matrix = trans_matrix.view((emb_size, -1))  # [emb_size, num_labels * emb_size]

  vector_com_logits = torch.matmul(vector_set_1, trans_matrix)
  vector_com_logits = vector_com_logits.view((batch_size, seq_length * num_labels, emb_size)) # [batch_size, seq_length*num_labels, emb_size]

  com_logits = torch.matmul(vector_com_logits, vector_set_2.permute((0, 2, 1)))  # [batch_size, seq_length*num_labels, seq_length]

  return com_logits.view((batch_size, seq_length, num_labels, seq_length))

def biaffine_classifier(start_logits, end_logits, num_labels, trans_matrix, add_bias_1=False, add_bias_2=False):
  com_logits = biaffine_mapping(vector_set_1=start_logits, 
                                vector_set_2=end_logits, 
                                num_labels=num_labels, 
                                trans_matrix=trans_matrix, 
                                add_bias_1=add_bias_1,
                                add_bias_2=add_bias_2)
  com_logits = com_logits.permute((0, 1, 3, 2)) # [batch_size, seq_length, seq_length, num_labels]
  return com_logits

def maxpool_hirchc_decode_algorithm(hiddens, 
                                    start_labels, 
                                    end_labels, 
                                    valid_second_id, 
                                    valid_category_id, 
                                    hirchc_elemement_info,
                                    valid_seq_lengths,
                                    max_element_length,
                                    second_classifiers,
                                    attention_mask=None,
                                    context_part='only_self',
                                    dropout_layer=None):
  batch_element_list = []
  for batch_idx, (example_start_labels, example_end_labels) in enumerate(zip(start_labels, end_labels)):
    example_hiddens = hiddens[batch_idx]
    element_list = []
    for start_idx, start_label in enumerate(example_start_labels):
      if start_label <= valid_category_id: continue
      end_idx = start_idx
      while end_idx < start_idx + max_element_length and \
          end_idx < valid_seq_lengths[batch_idx] and example_end_labels[end_idx] != start_label:
        end_idx = end_idx + 1
      if end_idx < valid_seq_lengths[batch_idx] and example_end_labels[end_idx] == start_label:
        first_label_id, second_label_id = start_label, 0
        if first_label_id > valid_second_id:
          entity_hiddens = maxpool_context_combination(example_hiddens=example_hiddens, 
                                                       start_id=start_idx, 
                                                       end_id=end_idx, 
                                                       mask=attention_mask[batch_idx], 
                                                       context_part=context_part,
                                                       dropout_layer=dropout_layer)
          second_logits = second_classifiers[first_label_id](entity_hiddens)
          second_label_id = second_logits.argmax(dim=0)
        element_name = hirchc_elemement_info.convert_second_label_element_tag(first_label_id=first_label_id, 
                                                                              second_label_id=second_label_id)
        element_list.append(EvalElement(value="", elem_name=element_name, start_id=start_idx, end_id=end_idx))
    batch_element_list.append(element_list)
  return batch_element_list

def maxpool_context_combination(example_hiddens, 
                                start_id, 
                                end_id, 
                                mask=None, 
                                context_part='only_self',
                                dropout_layer=None):
  '''
    Args:
      example_hiddens: [seq_length, hidden_size]
  '''
  start_hiddens, end_hiddens = example_hiddens[start_id], example_hiddens[end_id]
  element_hiddens_list = [start_hiddens, end_hiddens, 
                          start_hiddens + end_hiddens,
                          start_hiddens * end_hiddens]
  (seq_length) = example_hiddens.size()[0]      
  if mask is not None:
    seq_length = sum(mask)
  if context_part != 'only_self':
    example_hiddens = example_hiddens[: seq_length]
    left_hiddens = example_hiddens[: start_id]
    rigth_hiddens = example_hiddens[end_id + 1: ]
    left_context= left_hiddens.max(dim=0)[0]
    right_context = rigth_hiddens.max(dim=0)[0]
  # left_context = F.max_pool1d(input=left_hiddens.permute(1, 0), 
  #                             kernel_size=left_hiddens.size()[0]).view(-1)
  # right_context = F.max_pool1d(input=rigth_hiddens.permute(1, 0),
  #                              kernel_size=rigth_hiddens.size()[0]).view(-1)
  # if context_part == 'only_self': 
  #   # do nothing
  if context_part == 'all_flattern' or context_part == 'all_mean':
    element_hiddens_list = [left_context] + element_hiddens_list + [right_context]
  elif context_part == 'left_subset':
    element_hiddens_list = [left_context] + element_hiddens_list
  elif context_part == 'right_subset':
    element_hiddens_list = element_hiddens_list + [right_context]
  # print('###################################')
  # for a in element_hiddens_list:
  #   print(a.size())
  # element_hiddens = F.tanh(torch.stack(element_hiddens_list, dim=0))
  element_hiddens = F.tanh(torch.cat(element_hiddens_list, dim=0))
  if dropout_layer is not None:
    element_hiddens = dropout_layer(element_hiddens)
  return element_hiddens

def integrate_left_rigth_context(example_hiddens, start_id, end_id, dropout_layer=None, mask=None):
  left_context = example_hiddens[: start_id]                # [seq_length, hidden_dim]
  right_context = example_hiddens[end_id + 1: ]             
  element_context = example_hiddens[start_id: end_id + 1]

  max_left_context = left_context.max(dim=0)[0]
  max_right_context = right_context.max(dim=0)[0]
  max_element_context = element_context.max(dim=0)[0]       # [hidden_dim]

  element_logits = torch.cat([max_left_context, max_element_context, max_right_context])
  if dropout_layer is not None:
    element_logits = dropout_layer(element_logits)
  return element_logits

def clones(module, N):
  """Produce N identical layers."""
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    """Take in model size and number of heads."""
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0
    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    """Implements Figure 2"""
    if mask is not None:
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)

    query, key, value = [
      l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
      for l, x in zip(self.linears, (query, key, value))
    ]

    x, self.attn = attention(query, key, value, mask=mask,
                              dropout=self.dropout)

    x = x.transpose(1, 2).contiguous() \
        .view(nbatches, -1, self.h * self.d_k)

    return self.linears[-1](x)

def attention(query, key, value, mask=None, dropout=None):
  """Compute 'Scaled Dot Product Attention'"""
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    # print('#######################')
    # print('scores: ', scores.size())
    # print('mask:   ', mask.size())
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = F.softmax(scores, dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn

class MultiNonLinearClassifier(nn.Module):
  def __init__(self, hidden_size, num_label, dropout_rate):
    super(MultiNonLinearClassifier, self).__init__()
    self.num_label = num_label
    self.classifier1 = nn.Linear(hidden_size, hidden_size)
    self.classifier2 = nn.Linear(hidden_size, num_label)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, input_features):
    features_output1 = self.classifier1(input_features)
    # features_output1 = F.relu(features_output1)
    features_output1 = F.gelu(features_output1)
    features_output1 = self.dropout(features_output1)
    features_output2 = self.classifier2(features_output1)
    return features_output2

class DiceLoss(nn.Module):
  def __init__(self,
               smooth=1e-8,
               square_denominator=False,
               with_logits=True,
               reduction="mean"):
    super(DiceLoss, self).__init__()

    self.reduction = reduction
    self.with_logits = with_logits
    self.smooth = smooth
    self.square_denominator = square_denominator

  def forward(self,
              input,
              target,
              mask=None):

    flat_input = input.view(-1)
    flat_target = target.view(-1)

    if self.with_logits:
      flat_input = torch.sigmoid(flat_input)

    if mask is not None:
      mask = mask.view(-1).float()
      flat_input = flat_input * mask
      flat_target = flat_target * mask

    interection = torch.sum(flat_input * flat_target, -1)
    if not self.square_denominator:
      return 1 - ((2 * interection + self.smooth) /
                  (flat_input.sum() + flat_target.sum() + self.smooth))
    else:
      return 1 - ((2 * interection + self.smooth) /
                  (torch.sum(torch.square(flat_input,), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))



