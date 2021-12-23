import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torchcrf import CRF
from torch.autograd import Variable
from torch.autograd import Function
from dice import DiceLoss
from transformers import BertForQuestionAnswering

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, 100)
        self.classifier2 = nn.Linear(100, num_label)
        nn.init.xavier_uniform_(self.classifier1.weight)
        nn.init.constant_(self.classifier1.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier2.weight)
        nn.init.constant_(self.classifier2.bias, 0.0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)   
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

class GradReverseLayerFunction(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss

def SKL(logit, target, epsilon=1e-8):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    #bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    return (p* (rp- ry) * 2).sum()

class BertForQuestionAnswering(nn.Module):

    def __init__(self, bert_model, num_labels=2, dropout_prob=0.1, hidden_size=768):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = bert_model
        self.hidden_size = hidden_size # hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = num_labels
        self.span_embedding = MultiNonLinearClassifier(self.hidden_size * 2, 1, self.dropout_prob)
        self.label_embedding = Variable(torch.rand(size=(self.num_labels, self.hidden_size)), requires_grad=True).to("cuda")
        
        self.start_outputs = nn.Linear(self.hidden_size, 1)
        self.end_outputs = nn.Linear(self.hidden_size, 1)
        self.span_loss_candidates = "pred_and_gold"
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.dice_loss = DiceLoss(with_logits=True, smooth=1e-8)
        nn.init.xavier_uniform_(self.start_outputs.weight)
        nn.init.constant_(self.start_outputs.bias, 0.0)
        nn.init.xavier_uniform_(self.end_outputs.weight)
        nn.init.constant_(self.end_outputs.bias, 0.0)
        

    def forward(self, input_ids, 
                    attention_mask, 
                    token_type_ids=None,
                    start_labels=None, 
                    end_labels=None,
                    span_labels=None, seq_output=None, output_labels=None):

        # sequence_heatmap, _ = self.bert(input_ids=input_ids,
        #                             token_type_ids=token_type_ids,
        #                             attention_mask=attention_mask,
        #                             output_all_encoded_layers=False)
        if seq_output is None:
            sequence_heatmap, _, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            sequence_heatmap = seq_output
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        output_log_probs = self.label_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        c = output_log_probs[torch.arange(batch_size), output_labels, :].unsqueeze(1)
        c = c.repeat(1, seq_len, 1)
        sequence_heatmap = sequence_heatmap + c

        
        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)

        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        output, loss = {}, None
        if start_labels is not None:
            start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits, 
                                                           end_logits=end_logits, 
                                                           span_logits=span_logits, 
                                                           attention_mask=attention_mask, 
                                                           start_labels=start_labels,
                                                           end_labels=end_labels,
                                                           span_labels=span_labels)
            loss = 0.4762*start_loss + 0.4762*end_loss + 0.0476*match_loss
        output['loss'] = loss
        output['start_logits'] = start_logits
        output['end_logits'] = end_logits
        output['span_logits'] = span_logits
        return output

    def compute_loss(self, start_logits, 
                         end_logits, 
                         span_logits, 
                         attention_mask, 
                         start_labels,
                         end_labels,
                         span_labels):
        (batch_size, seq_len)= start_labels.size()
        start_float_label_mask = attention_mask.view(-1).float()
        end_float_label_mask = attention_mask.view(-1).float()
        match_label_row_mask = attention_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = attention_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)
        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                    )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        # print(start_labels)
        # start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        # start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        # end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        # end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        # match_loss = self.bce_loss(span_logits.view(batch_size, -1), span_labels.view(batch_size, -1).float())
        # match_loss = match_loss * float_match_label_mask
        # match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
       

        start_loss = self.dice_loss(start_logits, start_labels.float(), start_float_label_mask)
        end_loss = self.dice_loss(end_logits, end_labels.float(), end_float_label_mask)
        match_loss = self.dice_loss(span_logits, span_labels.float(), float_match_label_mask)

        return start_loss, end_loss, match_loss

        
        
class BertForSequenceClassification(nn.Module):
    def __init__(
        self,
        bert_model,
        num_classes,
        adv_alpha=1,
        dropout_prob=0.1,
        adv_by_gold_label=False,
        hidden_dim=768
    ):
        super(BertForSequenceClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bert = bert_model
        self.noise_var = 1e-5
        self.adv_step_size = 1e-3
        self.noise_gamma = 1e-6
        self.adv_alpha = adv_alpha
        self.project_norm_type = "inf"
        self.dropout = nn.Dropout(dropout_prob)
        self.adv_by_gold_label = adv_by_gold_label
        self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.bow_classifier = nn.Linear(self.hidden_dim, self.num_classes) # bag of word classfier
        # initialize the parameters
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.xavier_uniform_(self.bow_classifier.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0.0)
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.constant_(self.bow_classifier.bias, 0.0)

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def forward(self, input_ids, attention_mask, inputs_embeds=None):
        """ forward pass to classify the input tokens
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            output logits 
        """
        bert_seq_output, pooled_cls_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)[:2]
        # pooled_cls_output is batch_size x 746

        pooled_cls_output = self.projection(pooled_cls_output)
        output = self.dropout(pooled_cls_output)
        output = self.classifier(output)
        # output is batch_size x num_classes
        return output, bert_seq_output


    def neg_log_likelihood(self, input_ids, attention_mask, labels, return_output=False):
        """[summary]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        labels : torch.[cuda?].LongTensor
            correct labels, shape --> batch_size (or batch_size x 1)
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            the loss value (averaged over the batch)
        """
        output, seq_output = self.forward(input_ids, attention_mask) 
        cross_entropy_loss = nn.CrossEntropyLoss()

        loss = cross_entropy_loss(output, labels)
        
        embed = self.bert.get_input_embeddings()(input_ids)
        noise = embed.data.new(embed.size()).normal_(0, 1) * self.noise_var
        noise.requires_grad_()

        newembed = embed.data.detach() + noise
        newembed = newembed.reshape(list(input_ids.shape) + [-1])
        adv_logits, _ = self.forward(input_ids=None, inputs_embeds=newembed, attention_mask=attention_mask)
        if self.adv_by_gold_label:
            labels_onehot = F.one_hot(1-labels, 2)
            adv_loss = KL(adv_logits, labels_onehot.detach(), reduction="batchmean") 
        else:
            adv_loss = KL(adv_logits, output.detach(), reduction="batchmean") 
 
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        if (torch.isnan(norm) or torch.isinf(norm)):
            # skim this batch
            if return_output:
                output = torch.argmax(output, dim=-1)
            return loss, output
       
        if self.adv_by_gold_label:
            noise = noise + delta_grad * self.adv_step_size
        else:
            noise = noise + delta_grad * self.adv_step_size
        
        noise = self.adv_project(noise, norm_type=self.project_norm_type, eps=self.noise_gamma)
        noise_1 = embed.data.new(embed.size()).normal_(0, 1) * self.noise_var
        noise_2 = embed.data.new(embed.size()).normal_(0, 1) * self.noise_var
        newembed = embed.data.detach() + noise
        newembed = newembed.detach()

        newembed_1 = embed.data.detach() + noise_1
        newembed_1 = newembed_1.detach()

        newembed_2 = embed.data.detach() + noise_2
        newembed_2 = newembed_2.detach()
        # d_loss = 0.1*self.simsiam(newembed_1, newembed_2, attention_mask)
       
        adv_logits, _ = self.forward(input_ids=None, inputs_embeds=newembed, attention_mask=attention_mask)
        
        adv_loss_f = KL(adv_logits, output.detach())
        adv_loss_b = KL(output, adv_logits.detach())
        adv_loss = (adv_loss_f + adv_loss_b) * self.adv_alpha
        loss = loss + adv_loss #+ d_loss
        
        
        if return_output:
            output_label = torch.argmax(output, dim=-1)
            return loss, output_label, seq_output

        return loss
    
    def D_loss(self, p, z):
        # z = z.detach()
        # p = p / torch.norm(p, p=2, dim=1).unsqueeze(1)
        # z = z / torch.norm(z, p=2, dim=1).unsqueeze(1)
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p*z.detach()).sum(dim=1).mean()

    def simsiam(self, x1, x2, attention_mask):
        '''
        x1: augment(x1) x2 = augment(x2)
        '''
        _, z1, _ = self.bert(input_ids=None, attention_mask=attention_mask, inputs_embeds=x1)
        _, z2, _ = self.bert(input_ids=None, attention_mask=attention_mask, inputs_embeds=x2)
        # pooled_cls_output is batch_size x 746

        # print(pooled_cls_output)
        # output = self.dropout(pooled_cls_output)
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        d_loss = self.D_loss(p1, z2) / 2.0 + self.D_loss(p2, z1) / 2.0

        # output = self.classifier(output)
        
        return d_loss
    def bow_neg_log_likelihood(self, input_ids, attention_mask, labels, return_output=False):
        """[summary]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        labels : torch.[cuda?].LongTensor
            correct labels, shape --> batch_size (or batch_size x 1)
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            the loss value (averaged over the batch)
        """
        output = self.bert.embeddings.word_embeddings(input_ids)
        # output is batch_size x max_len x 768

        # remove the contributions from 0 positions as per attention mask
        output = torch.einsum("bld, bl->bld", output, attention_mask)

        pooled_output = torch.sum(output, dim=1) 
        # pooled_output is batch_size x 746

        output = self.bow_classifier(pooled_output)

        # forward(input_ids, attention_mask) 
        cross_entropy_loss = nn.CrossEntropyLoss()
        # the default reduction is mean
        # NOTE: there is no need to ignore any output... there is exactly one output per sentence
        loss = cross_entropy_loss(output, labels)
        
        if return_output:
            output = torch.argmax(output, dim=-1)
            return loss, output

        return loss


    def kl_divergence_loss(self, input_ids, attention_mask, tags):
        """computes the kl-divergence loss, i.e. 
        KL(P || Q) where P is expected output distribution (uniform over gold tags, 0 otherwise)
        and Q is the average [CLS] attention across all heads
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        tags : torch.[cuda?].LongTensor
            tags for which the attention is supposed to be high, 1 for those tokens, 0 otherwise 
            shape --> batch_size x max_seq_len 
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            the loss value (averaged over the batch)
        """
        _, _, attentions = self.bert(input_ids, attention_mask=attention_mask)
        #NOTE: I assume that the attentions corresponding to 0s in attention_mask is 0.0
        #TODO: confirm this case

        # normalize the tags
        updated_tags = tags + 1e-9
        normalizing_const = torch.sum(updated_tags, dim=1)
        normalized_tags = torch.einsum('ij,i->ij', updated_tags, 1.0/normalizing_const)

        # attentions is a tuple of 12 (layers), 
        # each of shape --> batch_size x 12 (for heads) x max_seq_len x max_seq_len
        last_layer_attention = attentions[-1]
        last_layer_CLS_attention = last_layer_attention[:, :, 0, :]
        # shape of last_layer_CLS_attention --> batch_size x 12 x max_seq_len
        last_layer_CLS_attention = last_layer_CLS_attention.permute(0, 2, 1)
        # shape of last_layer_CLS_attention --> batch_size x max_seq_len x 12        
        last_layer_CLS_attention_avg = torch.mean(last_layer_CLS_attention, dim=-1)
        last_layer_CLS_attention_avg_log = torch.log(last_layer_CLS_attention_avg + 1e-9)

        kld_loss = nn.KLDivLoss(reduction='batchmean')  # by default the reduction is 'mean'
        loss = kld_loss(last_layer_CLS_attention_avg_log, normalized_tags)

        return loss


    def get_top_tokens(self, input_ids, attention_mask, k=10):
        """ get the top k% tokens as per attention scores

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        k : int, optional
            the k in top-k, by default 10

        Returns
        -------
        List[List[int]]
            a binary footprint of what is selected in the top-k (marked by 1, others as 0) 
        """
        _, _, attention = self.bert(input_ids, attention_mask)
        # for each layer attention is (batch_size, num_heads, sequence_length, sequence_length)
        last_layer_attention = attention[-1]
        
        # aggregate all the attention heads by mean 
        last_layer_mean_attention = torch.mean(last_layer_attention, dim=1)
        # last_layer_mean_attention is (batch_size, sequence_length, sequence_length)

        last_layer_mean_CLS_attention = last_layer_mean_attention[:, 0, :]
        # last_layer_mean_CLS_attention is (batch_size, sequence_length)

        # NOTE: at this point converting to lists also in not a big time burden 
        
        output = []

        for i in range(len(last_layer_mean_CLS_attention)):

            line_len = len(input_ids[i].nonzero())  # including [CLS] and [SEP]

            # shape of score is line_len
            score = last_layer_mean_CLS_attention[i][:line_len]
            token_ids = input_ids[i][:line_len]

            num_words = int(0.01 * k * len(token_ids))
            if num_words == 0:
                # should at least contain a word 
                num_words = 1

            selected_indices = torch.argsort(score, descending=True)[:num_words]
            top_k_mask = [1.0 if i in selected_indices else 0.0 for i in range(len(token_ids))]
            output.append(top_k_mask)

        return output


class BertCRF(nn.Module):
    """ A CRF model to generate the I-O tags for rationales in input examples
        
        Parameters
        ----------
        bert_model : BertModel
            An instance of the pretrained bert model
        start_label_id : int
            The index of the <START TAG>
        stop_label_id : int
            The index of the <STOP TAG>
        num_labels : int
            number of output labels
        batch_size : int
            batch size of the input sequences, by defualt 32
        dropout_prob: float
            dropout probability, by default 0.2
    """

    def __init__(
            self, 
            bert_model, 
            start_label_id, 
            stop_label_id, 
            num_labels, 
            dropout_prob=0.1,
            match_loss_weight = 0.1,
            num_classes=None,
            bert_features_dim=768,
            without_label_embedding=False,
        ):
        super(BertCRF, self).__init__()
        self.bert_features_dim = bert_features_dim
        self.attention_features_dim = 12 # corresponding to the number of heads
        self.label_features_dim = 1 
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.bert = bert_model 
        self.without_label_embedding = without_label_embedding
        self.crf = CRF(num_labels, batch_first=True)
        # trans = allowed_transitions({0:'START', 1:'END', 2:'0', 3:'1']}, include_start_end=True)
        # self.crf = ConditionalRandomField(num_labels, include_start_end_trans=True, allowed_transitions=trans)
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        self.embedding_dim_label = bert_features_dim
        self.num_classes = num_classes
        self.match_loss_weight = match_loss_weight

        self.bert_features_to_label = nn.Linear(self.bert_features_dim, self.num_labels)
        self.double_bert_features_to_label = nn.Linear(2*self.bert_features_dim, self.num_labels)
        self.individual_bert_features_to_label = nn.Linear(self.bert_features_dim, self.num_labels)
        self.log_normalized_bert_features_to_label = nn.Linear(self.bert_features_dim, self.num_labels)
        # NOTE: one can share the individual feature and regular features weights 
        self.attention_features_to_label = nn.Linear(self.attention_features_dim, self.num_labels)
        self.avg_attention_features_to_label = nn.Linear(1, self.num_labels) # avg attn features
        self.label_features_to_label = nn.Linear(self.label_features_dim, self.num_labels)

        self.bow_features_to_label = nn.Linear(self.label_features_dim, self.embedding_dim_label)

        self.span_embedding = MultiNonLinearClassifier(self.bert_features_dim * 2, 1, self.dropout_prob)
        self.span_loss_candidates = "gold" #"pred_and_gold"
        self.label_embedding = Variable(torch.rand(size=(self.num_labels, self.bert_features_dim)), requires_grad=True).to("cuda")
        
        self.label_embedding_features_to_label = nn.Linear(self.bert_features_dim, self.num_labels)
        self.dice_loss = DiceLoss(with_logits=True, smooth=1e-8)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        # NOTE: one can inspect the label_features_to_label weights for sanity checking

        # initialize the parameters
        nn.init.xavier_uniform_(self.bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.double_bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.individual_bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.log_normalized_bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.attention_features_to_label.weight)
        nn.init.xavier_uniform_(self.avg_attention_features_to_label.weight)
        nn.init.xavier_uniform_(self.label_features_to_label.weight)
        nn.init.xavier_uniform_(self.bow_features_to_label.weight)
        nn.init.xavier_uniform_(self.label_embedding_features_to_label.weight)
        nn.init.constant_(self.bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.double_bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.individual_bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.log_normalized_bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.attention_features_to_label.bias, 0.0)
        nn.init.constant_(self.avg_attention_features_to_label.bias, 0.0)
        nn.init.constant_(self.label_features_to_label.bias, 0.0)
        nn.init.constant_(self.bow_features_to_label.bias, 0.0)
        nn.init.constant_(self.label_embedding_features_to_label.bias, 0.0)

        is_cuda = torch.cuda.is_available()

        self.float_type = torch.FloatTensor
        self.long_type = torch.LongTensor
        self.byte_type = torch.ByteTensor

        if is_cuda:
            self.float_type = torch.cuda.FloatTensor 
            self.long_type = torch.cuda.LongTensor
            self.byte_type = torch.cuda.ByteTensor

    def forward(
        self, 
        input_ids, 
        attention_mask,
        label_ids=None,
        span_labels=None,
        include_bert_features=False,
        include_double_bert_features=False,
        include_log_normalized_bert_features=False,
        include_attention_features=False,
        include_avg_attention_features=False,
        include_individual_bert_features=False,
        include_label_features=False,
        include_bow_features=False,
        include_label_embedding_features=False,
        classifier=None,
        bow_classifier=None,
        output_labels = None,
        output_logits = None,
        start_labels=None,
        end_labels=None
    ):
        """ forward pass of the model class
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input ids of sentences. shape ==> batch_size x max_seq_len 
        attention_mask : [type]
            [description]
        include_bert_features : bool, optional
            [description], by default False
        include_attention_features : bool, optional
            [description], by default False
        include_bert_features : bool, optional
            [description], by default False
        include_attention_features : bool, optional
            [description], by default False
        
        Returns
        -------
        List[List[int]]
            best path...
        """

        feats, match_loss_f = self._get_all_features(input_ids, 
            attention_mask,
            span_labels,
            include_bert_features,
            include_double_bert_features,
            include_log_normalized_bert_features,
            include_attention_features,
            include_avg_attention_features,
            include_individual_bert_features,
            include_label_features,
            include_bow_features,
            include_label_embedding_features,
            classifier,
            bow_classifier,
            output_labels,
            output_logits,
            start_labels=start_labels,
            end_labels=end_labels
        ) # b, l, 4


        mask = attention_mask.type(self.byte_type)
        if label_ids is None:
            return self.crf.decode(feats, mask=mask)
        else:
            log_likelihood = self.crf(feats, label_ids, mask=mask, reduction='token_mean') 
            if match_loss_f is not None:
                # print("loss: {}, crf loss: {}, match loss: {}".format(-log_likelihood+0.1*match_loss_f, -log_likelihood, 0.1*match_loss_f))
                return -log_likelihood+self.match_loss_weight*match_loss_f
            else:
                return -log_likelihood

    def print_weights(self,
        ITER,
        include_avg_attention_wts=False,
        include_bow_wts=False
    ):
        """ Print transition and other weights

        Parameters
        ----------
        ITER : int
            epoch count
        include_avg_attention_wts : bool, optional
            whether to print attention weights, by default False
        include_bow_wts : bool, optional
            whether to print bow weights, by default False
        """

        O_tag = 2
        I_tag = 3

        transitions = self.crf.transitions

        print("[transitions] ITER: %d I-I: %.2f O-O: %.2f I-0 %.2f 0-I %.2f" % (
            ITER,
            transitions[I_tag][I_tag].item(),
            transitions[O_tag][O_tag].item(),
            transitions[I_tag][O_tag].item(),
            transitions[O_tag][I_tag].item(),
        ))

        if include_avg_attention_wts:
            print("[attention weights] ITER: %d W_+: %.2f b_+: %.2f W_ %.2f b_ %.2f" % (
                ITER,
                self.avg_attention_features_to_label.weight.data[1].item(),
                self.avg_attention_features_to_label.bias.data[1].item(),
                self.avg_attention_features_to_label.weight.data[0].item(),
                self.avg_attention_features_to_label.bias.data[0].item(),
            ))

        if include_bow_wts:
            print("[bow weights] ITER: %d W_+: %.2f b_+: %.2f W_ %.2f b_ %.2f" % (
                ITER,
                self.bow_features_to_label.weight.data[1].item(),
                self.bow_features_to_label.bias.data[1].item(),
                self.bow_features_to_label.weight.data[0].item(),
                self.bow_features_to_label.bias.data[0].item(),
            ))
        
        return 

    def compute_loss(self,
                        span_logits, 
                        attention_mask, 
                        span_labels, start_labels=None, end_labels=None):
        (batch_size, seq_len)= attention_mask.size()
        match_label_row_mask = attention_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = attention_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)
        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            # start_preds = start_logits > 0
            # end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                    )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), span_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
        # match_loss = self.dice_loss(span_logits, span_labels.float(), float_match_label_mask)

        return match_loss

    def _get_all_features(
        self,
        input_ids, 
        attention_mask,
        span_labels=None,
        include_bert_features=False,
        include_double_bert_features=False,
        include_log_normalized_bert_features=False,
        include_attention_features=False,
        include_avg_attention_features=False,
        include_individual_bert_features=False,
        include_label_features=False,
        include_bow_features=False,
        include_label_embedding_features=False,
        classifier=None,
        bow_classifier=None,
        output_labels = None,
        output_logits = None,
        start_labels=None,
        end_labels=None,
    ):
        batch_size, max_seq_len = input_ids.shape

        # init emission features
        feats = torch.zeros((batch_size, max_seq_len, self.num_labels)).type(self.float_type)

        if include_bert_features:
            bert_feats = self._get_bert_features(input_ids, attention_mask)
            feats += bert_feats

        if include_double_bert_features:
            if output_labels is None:
                raise Exception("Need to have the output labels to get the double bert features")
            double_bert_feats = self._get_double_bert_features(input_ids, attention_mask, output_labels, output_logits)
            feats += double_bert_feats

        if include_log_normalized_bert_features:
            log_normalized_bert_features = self._get_log_normalized_bert_features(input_ids, attention_mask)
            feats += log_normalized_bert_features
        
        if include_attention_features:
            attention_features = self._get_attention_features(input_ids, attention_mask)
            feats += attention_features

        if include_avg_attention_features:
            avg_attention_features = self._get_avg_attention_features(input_ids, attention_mask)
            feats += avg_attention_features

        if include_individual_bert_features:
            individual_bert_features = self._get_individual_bert_features(input_ids, attention_mask)
            feats += individual_bert_features
        
        if include_label_features:

            # check if we have the classifier and the output labels
            if classifier is None or output_labels is None:
                raise Exception("Need to pass the classfier and a tensor specifying the outputs")

            label_features = self._get_label_features(input_ids, attention_mask, classifier, 
                                                        output_labels)
            feats += label_features


        if include_bow_features:
            if bow_classifier is None or output_labels is None:
                raise Exception("Need to pass the classfier and a tensor specifying the outputs")

            bow_feats = self._get_bow_features(input_ids, attention_mask, bow_classifier, 
                                                        output_labels)
            feats += bow_feats
        match_loss_f = None
        if include_label_embedding_features:
            if output_labels is None:
                raise Exception("Need to have the output labels to get the label embedding features")
            # double_bert_feats = self._get_double_bert_features(input_ids, attention_mask, output_labels, output_logits)
            label_embedding_feats, match_loss_f = self._get_label_embedding_features(input_ids, attention_mask, output_labels, span_labels=span_labels, start_labels=start_labels, end_labels=end_labels)
            feats += label_embedding_feats
        
        return feats, match_loss_f


    def _get_attention_features(self, input_ids, attention_mask):
        """ get features from BERT's attention 

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        _, _, attentions = self.bert(input_ids, attention_mask=attention_mask)

        # attentions is a tuple of 12 (layers), 
        # each of shape --> batch_size x 12 (for heads) x max_seq_len x max_seq_len
        last_layer_attention = attentions[-1]
        last_layer_CLS_attention = last_layer_attention[:, :, 0, :]
        # shape of last_layer_CLS_attention --> batch_size x 12 x max_seq_len
        last_layer_CLS_attention = last_layer_CLS_attention.permute(0, 2, 1)
        # shape of last_layer_CLS_attention --> batch_size x max_seq_len x 12        
        #NOTE: the attention features should be in log space 
        log_last_layer_CLS_attention = torch.log(last_layer_CLS_attention + 1e-9)
        attention_feats = self.attention_features_to_label(log_last_layer_CLS_attention)
        return attention_feats


    def _get_avg_attention_features(self, input_ids, attention_mask):
        """ get averaged features from BERT's attention

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        _, _, attentions = self.bert(input_ids, attention_mask=attention_mask)

        # attentions is a tuple of 12 (layers), 
        # each of shape --> batch_size x 12 (for heads) x max_seq_len x max_seq_len
        last_layer_attention = attentions[-1]
        last_layer_CLS_attention = last_layer_attention[:, :, 0, :]
        # shape of last_layer_CLS_attention --> batch_size x 12 x max_seq_len
        last_layer_CLS_attention = last_layer_CLS_attention.permute(0, 2, 1)
        # shape of last_layer_CLS_attention --> batch_size x max_seq_len x 12        
        last_layers_CLS_attention_avg = torch.mean(
            last_layer_CLS_attention, dim=-1).unsqueeze(dim=-1)

        # add log-features 
        last_layers_CLS_attention_avg_log = torch.log(last_layers_CLS_attention_avg + 1e-9) 
        avg_attention_feats = self.avg_attention_features_to_label(last_layers_CLS_attention_avg_log)
        return avg_attention_feats

    def _get_bert_features(self, input_ids, attention_mask):
        """ get features from BERT 

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)

        bert_feats = self.bert_features_to_label(bert_seq_out)

        return bert_feats


    def _get_double_bert_features(self, input_ids, attention_mask, output_labels, output_logits):
        """ get bert features such that the output label 1 are on the left, and 0 on the rigt

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        output_labels : torch.[cuda?].LongTensor
            output predictions (or even possibly ground truth)
            shape --> [batch_size]

        Returns
        -------
        torch.[cuda?].FloatTensor
            features ...
        """

        #NOTE: this only works for a binary classification task..

        
        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)
        # shape fo bert_seq_out is batch_size x max_seq_len x 746
    
        first_half = torch.einsum('ijk,i->ijk', bert_seq_out, output_logits[:, 0])
        second_half = torch.einsum('ijk,i->ijk', bert_seq_out, output_logits[:, 1])

        output = torch.cat((first_half, second_half), dim=-1)
    
        double_bert_feats = self.double_bert_features_to_label(output)

        return double_bert_feats

    def _get_label_embedding_features(self, input_ids, attention_mask, output_labels, span_labels, start_labels, end_labels):
        """ get bert features such that the output label 1 are on the left, and 0 on the rigt

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        output_labels : torch.[cuda?].LongTensor
            output predictions (or even possibly ground truth)
            shape --> [batch_size]

        Returns
        -------
        torch.[cuda?].FloatTensor
            features ...
        """
        '''
        # version #1
        u = self.dropout(self.batch_cosinesim(bert_seq_out, self.c))  # [b, l, k]
        # u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze())  # [b, l, k]
        u = u.reshape((batch_size, max_seq_len, -1))
        label_embedding_bert_feats = u
        # m = self.dropout(self.phrase_extract(u))  # [b, l, 1]
        # b = torch.softmax(m, dim=1)  # [b, l, 1]
        # batch_size, max_seq_len, self.num_labels
        # print("m: ", m.shape)
        # print(u)
        # label_embedding_bert_feats = self.label_embedding_features_to_label(m)
        return label_embedding_bert_feats
        
        '''
        bert_seq_out = self.bert(input_ids, attention_mask=attention_mask)[0]
        batch_size, seq_len = input_ids.shape
        
        # version #2
        output_log_probs = self.label_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        c = output_log_probs[torch.arange(batch_size), output_labels, :].unsqueeze(1)
        c = c.repeat(1, seq_len, 1)
        if not self.without_label_embedding:
            bert_seq_out = bert_seq_out + c

        label_embedding_bert_feats = self.label_embedding_features_to_label(bert_seq_out)
        if span_labels is not None:
            start_extend = bert_seq_out.unsqueeze(2).expand(-1, -1, seq_len, -1)
            end_extend = bert_seq_out.unsqueeze(1).expand(-1, seq_len, -1, -1)

            span_matrix = torch.cat([start_extend, end_extend], 3)
            span_logits = self.span_embedding(span_matrix).squeeze(-1)
            match_loss = self.compute_loss(span_logits, attention_mask, span_labels, start_labels, end_labels)
        else:
            match_loss = None
        return label_embedding_bert_feats, match_loss
    

    def _get_log_normalized_bert_features(self, input_ids, attention_mask):
        """ get features from BERT after first normalizing (softmax) and then log 

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)

        normalized_bert_seq_out = nn.functional.softmax(bert_seq_out, dim=-1)

        log_normalized_bert_seq_out = torch.log(normalized_bert_seq_out + 1e-9) 

        bert_feats = self.log_normalized_bert_features_to_label(log_normalized_bert_seq_out)

        return bert_feats



    def _get_individual_bert_output(self, input_ids, attention_mask):
        """ get **output** from BERT for individual tokens
        don't confuse with _get_individual_bert_features

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len

        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x bert_output_dim
        """
        batch_size, max_seq_len = input_ids.shape
        bert_seq_out = torch.zeros(batch_size, max_seq_len, self.bert_features_dim).type(
            self.float_type)

        for i in range(max_seq_len):
            # pass individual tokens...
            bert_output, _, _ = self.bert(input_ids[:, i].unsqueeze(dim=1),
                                            attention_mask[:, i].unsqueeze(dim=1))
            bert_seq_out[:, i, :] = bert_output[:, 0, :]

        return bert_seq_out



    def _get_individual_bert_features(self, input_ids, attention_mask):
        """ get features from BERT for individual tokens

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """

        bert_seq_out = self._get_individual_bert_output(input_ids, attention_mask)
        individual_bert_feats = self.individual_bert_features_to_label(bert_seq_out)

        return individual_bert_feats


    def _get_label_features(self, input_ids, attention_mask, classifier, output_labels):
        """get the label features... theta^T f_{BERT}(x_t)[y]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        classifier : nn.Module
            classifier from the Prediction Model to get the logits
        output_labels : torch.[cuda?].LongTensor
            the predicted output labels for the batch
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """

        batch_size = input_ids.shape[0]
        # individual_bert_features = self._get_individual_bert_output(input_ids, attention_mask)
        # shape of individual_bert_features is batch_size x max_seq_len x bert_feature_dim (746)

        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)
        # shape of bert_seq_out is batch_size x max_seq_len x bert_feature_dim (746)

        output_logits = classifier(bert_seq_out) 
        # shape of output_logits is  batch_size x max_seq_len x classification_labels

        output_log_probs = torch.log(nn.functional.softmax(output_logits, dim=-1) + 1e-9)

        #NOTE: the output label would not be the same for each sentence...
        output_log_probs_for_label = output_log_probs[torch.arange(batch_size), :, output_labels].unsqueeze(
            dim=-1)
        # shape of output_probs_for_label is  batch_size x max_seq_len x 1

        label_feats = self.label_features_to_label(output_log_probs_for_label) 
        # shape of label_feats is  batch_size x max_seq_len x num_labels

        return label_feats


    def _get_bow_features(self, input_ids, attention_mask, bow_classifier, output_labels):
        """get the label features... theta^T f_{bow}(embed(x_t))[y]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        bow_classifier : nn.Module
            classifier from the Prediction Model to get the logits
        output_labels : torch.[cuda?].LongTensor
            the predicted output labels for the batch
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """

        batch_size = input_ids.shape[0]

        output = self.bert.embeddings.word_embeddings(input_ids)
        # output is batch_size x max_len x 768

        # remove the contributions from 0 positions as per attention mask
        output = torch.einsum("bld, bl->bld", output, attention_mask)

        output_logits = bow_classifier(output) 
        # shape of output_logits is  batch_size x max_seq_len x classification_labels

        output_probs = nn.functional.softmax(output_logits, dim=-1)

        #NOTE: the output label would not be the same for each sentence...
        output_probs_for_label = output_probs[torch.arange(batch_size), :, output_labels].unsqueeze(
            dim=-1)
        # shape of output_probs_for_label is  batch_size x max_seq_len x 1

        # add log-features instead
        output_log_probs_for_label = torch.log(output_probs_for_label + 1e-9)

        bow_feats = self.bow_features_to_label(output_log_probs_for_label) 
        # shape of label_feats is  batch_size x max_seq_len x num_labels

        return bow_feats


    def batch_cosinesim(self, v, c):
        normalized_v = v / torch.norm(v, p=2, dim=2).unsqueeze(2).repeat(1, 1, self.bert_features_dim)
        normalized_c = c / torch.norm(c, p=2, dim=1).unsqueeze(1).repeat(1, self.bert_features_dim)
 
        # nan -> pad_idx(0) or not-aligned label part
        normalized_v[torch.isnan(normalized_v)] = 0  # [b, l, h]
        normalized_c[torch.isnan(normalized_c)] = 0  # [k, h]

        normalized_c = normalized_c.unsqueeze(0).repeat(normalized_v.shape[0], 1, 1).permute(0, 2, 1)  # [b,h,k]
        g = torch.bmm(normalized_v, normalized_c)
        return g