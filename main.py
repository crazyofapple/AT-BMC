import torch
import torch.optim as optim
from torch.utils import data
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import numpy as np
import argparse
from models import BertForSequenceClassification, BertCRF
from transformers import AdamW, get_linear_schedule_with_warmup
from data_utils import *
import time
import random
from sklearn.metrics import precision_recall_fscore_support
import os

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=123, type=int,
                        help = "random seed for torch, numpy, random")
    parser.add_argument('--epochs', default=5, type=int,
                        help = "number of training epochs")
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help = "learning rate")
    parser.add_argument('--data_dir', default="", type=str,
                        help = "data directory")
    parser.add_argument('--upper_case', action='store_true',
                        help="have tokenizer upper case (default is lower case)")
    parser.add_argument('--max_seq_len', type=int, default=32,
                        help="max sequence length")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="gradient_accumulation_steps")
    parser.add_argument('--extraction_coeff', type=float, default=1.0,
                        help='coefficient of extraction loss')
    parser.add_argument('--bow_coeff', type=float, default=0.0,
                        help='coefficient of extraction loss')
    parser.add_argument('--kld_coeff', type=float, default=0.0,
                        help='coefficient of kld loss')
    parser.add_argument('--fraction_rationales', type=float, default=1.0,
                        help='what fraction of sentences have rationales')
    parser.add_argument('--prediction_coeff', type=float, default=1.0,
                        help='coefficient of prediction loss')
    parser.add_argument('--weight_decay_finetune', type=float, default=1e-5,
                        help='weight decay finetune') 
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, 
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, 
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm.")
    parser.add_argument("--save_extraction_model", default=None, type=str,
                        help="path to save extraction model")
    parser.add_argument("--load_extraction_model", default=None, type=str,
                        help="path to load extraction model")
    parser.add_argument("--save_prediction_model", default=None, type=str,
                        help="path to save prediction model")
    parser.add_argument("--load_prediction_model", default=None, type=str,
                        help="path to load prediction model")
    parser.add_argument("--evaluate_every", default=1000, type=int,
                        help="evaluate every xx number of steps")
    parser.add_argument("--adv_alpha", default=1.0, type=float,
                        help="adv loss weight")
    parser.add_argument("--adv_by_gold_label", default=False, action="store_true")
    parser.add_argument("--without_label_embedding", default=False, action="store_true")
    parser.add_argument("--match_loss_weight", default=0.1, type=float,
                        help="match loss weight")
    parser.add_argument("--dropout_prob", default=0.1, type=float,
                        help="dropout prob")
    parser.add_argument("--irm_weight", default=0.0, type=float,
                        help="irm loss weight")
    parser.add_argument("--print_every", default=100, type=int,
                        help="print loss every xx number of steps")
    parser.add_argument("--attention_top_k", default=10, type=int,
                        help="top k in attention")
    parser.add_argument("--dataset", default="movie_reviews", type=str,
                        help="dataset")
    parser.add_argument("--include_attention_features", action='store_true',
                        help="whether to include attention features")
    parser.add_argument("--include_avg_attention_features", action='store_true',
                        help="whether to include average attention features")
    parser.add_argument("--include_bert_features", action='store_true',
                        help="whether to include BERT features")
    parser.add_argument("--include_double_bert_features", action='store_true',
                        help="whether to include double BERT features")
    parser.add_argument("--include_label_embedding_features", action='store_true',
                        help="whether to include label embedding features")
    parser.add_argument("--include_individual_bert_features", action='store_true',
                        help="whether to include individual BERT features")
    parser.add_argument("--include_log_normalized_bert_features", action='store_true',
                        help="whether to include log normalized BERT features")
    parser.add_argument("--include_label_features", action='store_true',
                        help="whether to include label features")
    parser.add_argument("--include_bow_features", action='store_true',
                        help="whether to include bow features")
    parser.add_argument("--dump_rationales", action='store_true',
                        help="whether to dump generated test rationales or not")
    parser.add_argument("--use_oracle_labels", action='store_true',
                        help="whether to use oracle label for extraction or not")
    return parser

def mrc_decode_elements(start_logits, 
                        end_logits, 
                        span_logits, 
                        attention_mask):
    # start_preds, end_preds = start_logits > 0, end_logits > 0
    # start_preds = start_preds.tolist()
    # end_preds = end_preds.tolist()
    start_logits = start_logits.detach().cpu().numpy()      # [batch, seq, nclasses]
    end_logits = end_logits.detach().cpu().numpy()          # [batch, seq, nclasses]
    span_logits = span_logits.detach().cpu().numpy()
    preds = []
    for inst_idx, (inst_start_logits, inst_end_logits, inst_span_logits, inst_mask) \
                in enumerate(zip(start_logits, end_logits, span_logits, attention_mask)):

        valid_length = sum(inst_mask.tolist())

        tag_list = [2] * valid_length # + [0] * max(0, tag_len - valid_length)
        tag_list[0] = 0
        tag_list[valid_length-1] = 1
        inst_start_labels = (inst_start_logits > 0).tolist()
        
        inst_end_labels = (inst_end_logits > 0).tolist()
        inst_span_labels = (inst_span_logits > 0).tolist()
        flag = False
        for start_id in range(valid_length):
            for end_id in range(start_id+1, valid_length):
                if inst_start_labels[start_id] & \
                    inst_end_labels[end_id] & inst_span_labels[start_id][end_id]:
                    # print("ok")
                    flag = True
                    inst_start_labels[start_id] = 0
                    inst_end_labels[end_id] = 0
                    tag_list[start_id:end_id+1] = [3] * (end_id-start_id+1)       
        preds.append(tag_list)
    return preds


def evaluate(prediction_model, extraction_model, predict_dataloader, positive_tag_idx,
                ITER, dataset_name, args):
    
    prediction_model.eval()
    extraction_model.eval()

    # tagging stats
    all_tag_preds = []
    all_tag_gold = []
    total_tags = 0.
    correct_tags = 0.

    # prediction stats
    correct_preds = 0.
    total_preds = 0.0
    all_pred_preds = []
    all_pred_gold = []

    # top k attention stats
    top_k_attention_preds = []
    top_k_attention_gold = []
    top_k_correct_preds = 0.0
    top_k_total_preds = 0.0

    start = time.time()
    with torch.no_grad():
        for step, batch in enumerate(predict_dataloader):

            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch

            # compute outputs from the prediction module
            pred_scores, output_logits = prediction_model(input_ids, attention_mask)
            pred_labels = torch.argmax(pred_scores, dim=-1)
            total_preds += pred_labels.shape[0]
            correct_preds += torch.sum(pred_labels == label_ids)
            all_pred_preds.extend(pred_labels.tolist())
            all_pred_gold.extend(label_ids.tolist())


            # compute outputs from extraction model

            # select only the ones with rationales
            input_ids = input_ids[has_rationale]
            attention_mask = attention_mask[has_rationale]
            tag_ids = tag_ids[has_rationale]
            pred_labels = pred_labels[has_rationale]
            pred_scores = pred_scores[has_rationale]
            output_logits = output_logits[has_rationale]
            if args.use_oracle_labels:
                labels_to_be_used = label_ids[has_rationale] # oracle labels
            else:
                labels_to_be_used = pred_labels # predicted labels
                logits_to_be_used = pred_scores



            if len(input_ids) > 0:

                gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                pred_tags = extraction_model(
                    input_ids,
                    attention_mask=attention_mask,
                    include_bert_features=args.include_bert_features,
                    include_double_bert_features=args.include_double_bert_features,
                    include_log_normalized_bert_features=args.include_log_normalized_bert_features,
                    include_attention_features=args.include_attention_features,
                    include_avg_attention_features=args.include_avg_attention_features,
                    include_individual_bert_features=args.include_individual_bert_features,
                    include_label_features=args.include_label_features,
                    include_bow_features=args.include_bow_features,
                    include_label_embedding_features=args.include_label_embedding_features,
                    classifier=prediction_model.classifier,
                    bow_classifier=prediction_model.bow_classifier,
                    output_labels=labels_to_be_used,
                    output_logits=logits_to_be_used
                )
                
                pred_tags_flat = [val for sublist in pred_tags for val in sublist]
                all_tag_preds.extend(pred_tags_flat)
                all_tag_gold.extend(gold_tags_flat)

                assert len(gold_tags_flat) == len(pred_tags_flat), print("{} vs {}".format(len(gold_tags_flat), len(pred_tags_flat)))

                total_tags += len(gold_tags_flat)
                correct_tags += np.sum(np.array(gold_tags_flat) == np.array(pred_tags_flat))

                if args.dump_rationales and dataset_name == 'test':
                    suffix = "bert_features=" + str(args.include_bert_features) + \
                        "_include_double_bert_features=" + str(args.include_double_bert_features) + \
                        "_attention_features=" + str(args.include_attention_features) + \
                        "_prediction_coeff=" + str(args.prediction_coeff) + \
                        "_extraction_coeff=" + str(args.extraction_coeff) + \
                        "_kld_coeff=" + str(args.kld_coeff) + "_test.txt"
                    attn_outfile = "top_k_attention_" + suffix
                    attn_outfile = os.path.join(args.data_dir, attn_outfile)
                    dump_rationales(input_ids, attn_outfile, attn_pred_tags, 1, args)
                    crf_outfile = "crf_" + suffix
                    crf_outfile = os.path.join(args.data_dir, crf_outfile)
                    dump_rationales(input_ids, crf_outfile, pred_tags, positive_tag_idx, args)



    # print prediction results ...
    precision_pred, recall_pred, f1_pred, _ = precision_recall_fscore_support(all_pred_gold, \
        all_pred_preds, average='micro')

    acc_pred = (correct_preds * 1.)/total_preds

    end = time.time()
    print('ITER: %d | dataset: %s | Predict Acc: %.2f | P: %.2f | R: %.2f | F1: %.2f | T: %.3f mins' \
        % (ITER, dataset_name, 100.*acc_pred, \
            100.*precision_pred, 100.*recall_pred, 100.*f1_pred, (end-start)/60.0))
    print('--------------------------------------------------------------')

    # print tagging results ...
    precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(
        np.array(all_tag_gold), np.array(all_tag_preds), labels=[positive_tag_idx])

    acc_tagging = (1. * correct_tags)/total_tags

    end = time.time()
    print('ITER: %d | dataset: %s | Tagging Acc: %.2f | P: %.2f | R: %.2f | F1: %.2f | T: %.3f mins' \
        % (ITER, dataset_name, 100.*acc_tagging, \
            100.*precision_tagging, 100.*recall_tagging, 100.*f1_tagging, (end-start)/60.0))
    print('--------------------------------------------------------------')


    return  {
            'p_pred': precision_pred,
            'r_pred': recall_pred,
            'f1_pred': f1_pred,
            'acc_pred': acc_pred,
            'p_tag': precision_tagging,
            'r_tag': recall_tagging,
            'f1_tag': f1_tagging,
            'acc_tag': acc_tagging,
            'p_attn': 0, #precision_attn,
            'r_attn': 0, #recall_attn,
            'f1_attn': 0, #f1_attn,
            'acc_attn': 0, #acc_top_k_attn,
        }


def dump_rationales(token_ids, outfile, pred_list, positive_idx, args, tokenizer=None):
    """ dump rationales as per a given extraction technique

    Parameters
    ----------
    token_ids : torch.[cuda?].LongTensor
        input token ids, shape --> batch_size x max_seq_len
    outfile : str
        output file path
    pred_list : list[list[int]]
        predicted tokens
    positive_idx : int
        the index of the positive tag "1" which indicates the selected token index
    args : [object?]
        parser arguments
    tokenizer : BertTokenizer, optional
        tokenizer for converting the ids to tokens, if None we will load the bert-tokenizer 

    Returns
    -------
    None
        just dumps the extracted rationales in the file
    """

    # init the tokenizer 
    if tokenizer is None:
        if args.upper_case:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    fw = open(outfile, 'w')  # overwrite

    for idx, t_ids in enumerate(token_ids):

        num_tokens = len(t_ids.nonzero())
        bp_tokens = tokenizer.convert_ids_to_tokens(t_ids[:num_tokens])

        # make sure we have the same number of byte-pair tokens as output predictions
        assert len(bp_tokens) == len(pred_list[idx])

        selected_idx = [1 if i == positive_idx else 0 for i in pred_list[idx]]

        output_words = ["**" + bp_tokens[i] + "**" if selected_idx[i] == 1 else bp_tokens[i] \
            for i in range(num_tokens)]

        output_line = " ".join(output_words)
        fw.write(output_line + "\n")

    fw.close()

    return None


def main():

    is_cuda = torch.cuda.is_available()
    float_type = torch.FloatTensor
    if is_cuda:
        float_type = torch.cuda.FloatTensor

    # parse arguments
    args = init_parser().parse_args()

    # set seed
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    if args.upper_case:
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if args.dataset == 'movie_reviews':
        dataset_processor = MovieReviewsProcessor()
    elif args.dataset == 'esnli':
        dataset_processor = EsnliProcessor()
    elif args.dataset == 'evinf':
        dataset_processor = EvinfProcessor()
    elif args.dataset == 'propaganda':
        dataset_processor = PropagandaProcessor()
    elif args.dataset == 'multirc':            # modified add
        dataset_processor = MultircProcessor() # modified add
    elif args.dataset == 'multi_rc':
        dataset_processor = MultiRCProcessor() # modified delete
    else:
        raise Exception("No (or wrong) dataset specified")


    # set fraction rationales
    dataset_processor.set_fraction_rationales(args.fraction_rationales)

    # get training/dev/test examples 
    train_examples = dataset_processor.get_train_examples(args.data_dir)
    dev_examples = dataset_processor.get_dev_examples(args.data_dir)
    test_examples = dataset_processor.get_test_examples(args.data_dir)

    # print (train_examples[0])

    tag_map = dataset_processor.get_tag_map()
    num_labels = dataset_processor.get_num_labels()
    num_tags = dataset_processor.get_num_tags()

    train_dataset = DatasetWithRationales(train_examples, tokenizer, tag_map, args.max_seq_len, \
        args.dataset)
    dev_dataset = DatasetWithRationales(dev_examples, tokenizer, tag_map, args.max_seq_len, \
        args.dataset)
    test_dataset = DatasetWithRationales(test_examples, tokenizer, tag_map, args.max_seq_len, \
        args.dataset)


    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        # pin_memory=True,
        collate_fn=DatasetWithRationales.pad,
        worker_init_fn=np.random.seed(args.seed),
    )

    dev_dataloader = data.DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn= DatasetWithRationales.pad,
        worker_init_fn=np.random.seed(args.seed),
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn= DatasetWithRationales.pad,
        worker_init_fn=np.random.seed(args.seed),
    )

    print("data completed.")
    # init the models
    if args.upper_case: #TODO: change the code to input model directory
        bert_model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
    else:
        bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    print("pre-trained model ok.")
    print("adv_alpha: ", args.adv_alpha)
    print("match_loss_weight: ", args.match_loss_weight)
    extraction_model = BertCRF(bert_model, start_label_id=tag_map.get('SEP', None), 
                                                stop_label_id=tag_map.get('SEP', None), num_labels=num_tags, 
                                                num_classes=num_labels, match_loss_weight=args.match_loss_weight,
                                                dropout_prob=args.dropout_prob, bert_features_dim=bert_model.config.hidden_size)
    # extraction_model = BertForQuestionAnswering(bert_model)
    prediction_model = BertForSequenceClassification(bert_model, num_classes=num_labels, 
        adv_alpha=args.adv_alpha,dropout_prob=args.dropout_prob, 
        adv_by_gold_label=args.adv_by_gold_label,
        hidden_dim=bert_model.config.hidden_size)
    
    is_cuda = torch.cuda.is_available()
    print("is cuda available: ", is_cuda)
    if is_cuda:
        prediction_model.cuda()
        extraction_model.cuda()

    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs


    # init the optimizer

    named_params = list(extraction_model.named_parameters()) + \
        list(prediction_model.classifier.named_parameters()) + \
        list(prediction_model.bow_classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params \
            if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_finetune},
        {'params': [p for n, p in named_params \
            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    global_step = 0

    best_valid_f1 = 0.0

    best_val_results = {
        'p_pred': 0.,
        'r_pred': 0.,
        'f1_pred': 0.,
        'acc_pred': 0.,
        'p_tag': 0.,
        'r_tag': 0.,
        'f1_tag': 0.,
        'acc_tag': 0.,
        'p_attn': 0.,
        'r_attn': 0.,
        'f1_attn': 0.,
        'acc_attn': 0.,
    }

    improvement_threshold = 10 # epochs 
    no_improvements_since = 0

    # zero the grads before training
    prediction_model.zero_grad()
    extraction_model.zero_grad()

    for ITER in range(args.epochs):
        ###############################################################################################################
        # training loop
        print("Training epoch: {}".format(ITER))
        st_time = time.time()
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
            

            prediction_model.train()
            extraction_model.train()

            classification_loss, output_labels, output_logits = prediction_model.neg_log_likelihood(input_ids,
                                attention_mask, label_ids, return_output=True)

            # bow_classification_loss = prediction_model.bow_neg_log_likelihood(input_ids,
            #                     attention_mask, label_ids, return_output=False)
            # classification_loss = 0
            bow_classification_loss = 0
            # select only the ones with rationales
            input_ids = input_ids[has_rationale]
            attention_mask = attention_mask[has_rationale]
            tag_ids = tag_ids[has_rationale]
            output_labels = label_ids[has_rationale] # change to golden label by dongfangli
            output_logits = output_logits[has_rationale]
            start_labels = start_labels[has_rationale]
            end_labels = end_labels[has_rationale]
            span_labels = span_labels[has_rationale]


            if len(input_ids) > 0:
                extraction_loss = extraction_model(
                    input_ids,
                    attention_mask=attention_mask,
                    label_ids=tag_ids,
                    span_labels=span_labels,
                    include_bert_features=args.include_bert_features,
                    include_double_bert_features=args.include_double_bert_features,
                    include_log_normalized_bert_features=args.include_log_normalized_bert_features,
                    include_attention_features=args.include_attention_features,
                    include_avg_attention_features=args.include_avg_attention_features,
                    include_individual_bert_features=args.include_individual_bert_features,
                    include_label_features=args.include_label_features,
                    include_bow_features=args.include_bow_features,
                    include_label_embedding_features=args.include_label_embedding_features,
                    classifier=prediction_model.classifier,
                    bow_classifier=prediction_model.bow_classifier,
                    output_labels=output_labels,
                    output_logits=output_logits,
                    start_labels=start_labels,
                    end_labels=end_labels
                )
                if args.kld_coeff != 0.0:
                    target_tags = (tag_ids == tag_map["1"]).type(float_type)
                    if 'propaganda' in args.dataset:
                        is_positive = (output_labels == 1)
                        kld_loss = prediction_model.kl_divergence_loss(
                            input_ids[is_positive],
                            attention_mask[is_positive],
                            target_tags[is_positive])
                    else:
                        kld_loss = prediction_model.kl_divergence_loss(
                            input_ids,
                            attention_mask,
                            target_tags)
                else:
                    kld_loss = torch.zeros(1, requires_grad=True).type(float_type)

            else:
                extraction_loss = torch.zeros(1, requires_grad=True).type(float_type)
                kld_loss = torch.zeros(1, requires_grad=True).type(float_type)

            loss = (args.extraction_coeff * extraction_loss) + \
                   (args.prediction_coeff * classification_loss) + \
                   (args.bow_coeff * bow_classification_loss) + \
                   (args.kld_coeff * kld_loss) 

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.print_every == 0:
                print ("ITER: %3d\tSTEP: %3d\tExtraction Loss: %.3f\tPrediction Loss: %.3f\tPrediction Loss (bow): %.3f\tKL Divergence Loss: %.3f\tTotal:%.2f"\
                %(
                    ITER,
                    step,
                    args.extraction_coeff * extraction_loss,
                    args.prediction_coeff * classification_loss,
                    args.bow_coeff*bow_classification_loss,
                    args.kld_coeff*kld_loss,
                    loss * args.gradient_accumulation_steps
                ))

            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(extraction_model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(prediction_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                prediction_model.zero_grad()
                extraction_model.zero_grad()
                global_step += 1
                if (step + 1) % args.evaluate_every == 0:
                    evaluate(prediction_model, extraction_model, dev_dataloader, tag_map["1"], \
                        ITER, "dev:step=" + str(step+1), args)


        et_time = time.time()
        print ("Training time = %.3f mins" %((et_time- st_time)/60.0))

        ###############################################################################################################
    

        # 太慢了，不评估
        # evaluate(prediction_model, extraction_model, train_dataloader, tag_map["1"], ITER, "train", 
        #     args)

        # p_pred, r_pred, f1_pred, acc_pred, p_tag, r_tag, f1_tag, acc_tag
        # prediction_model.load_state_dict(torch.load(os.path.join(args.save_prediction_model, "prediction_model.pt")))
        # extraction_model.load_state_dict(torch.load(os.path.join(args.save_extraction_model, "extraction_model.pt")))
        dev_results = evaluate(
            prediction_model, extraction_model, dev_dataloader, tag_map["1"], ITER, "dev", args)


        f1_combined = args.extraction_coeff * dev_results['f1_tag'] \
            + args.prediction_coeff * dev_results['f1_pred'] \
            + args.kld_coeff * dev_results['f1_attn']

        if f1_combined > best_valid_f1:
            print(ITER)
            best_valid_f1 = f1_combined
            if args.save_extraction_model is not None:
                torch.save(extraction_model.state_dict(), \
                    os.path.join(args.save_extraction_model, "extraction_model.pt"))
            if args.save_prediction_model is not None:
                torch.save(prediction_model.state_dict(), \
                    os.path.join(args.save_prediction_model, "prediction_model.pt"))
            # update the best values
            for k, v in dev_results.items():
                best_val_results[k] = v
            no_improvements_since = 0.0
            evaluate(prediction_model, extraction_model, test_dataloader, tag_map["1"], ITER, "test", \
                args)
        else:
            # no improvements
            no_improvements_since += 1.0

            # exit the training loop if the validation F1 hasn't improved since last few times
            if no_improvements_since >= improvement_threshold:
                break



    print ("best-dev-results\t%.2f\t%.2f\t%.2f\t%.2f\t\t%.2f\t%.2f\t%.2f\t%.2f\t\t%.2f\t%.2f\t%.2f\t%.2f" %(
        100. * best_val_results['p_pred'], 100. * best_val_results['r_pred'],\
        100. * best_val_results['f1_pred'], 100. * best_val_results['acc_pred'],\
        100. * best_val_results['p_tag'], 100. * best_val_results['r_tag'], \
        100. * best_val_results['f1_tag'], 100. * best_val_results['acc_tag'], \
        100. * best_val_results['p_attn'], 100. * best_val_results['r_attn'],\
        100. * best_val_results['f1_attn'], 100.* best_val_results['acc_attn'], \
    ))


    return

if __name__ == '__main__':
    # do I need this, now that num_workers = 0?
    # torch.multiprocessing.set_start_method('spawn') 
    main()
