import os
import torch
from torch.utils import data
import numpy as np
from tabulate import tabulate
import random
import torch.nn.functional as F

class InputFeatures(object):
    """A single set of features of data.
       Result of convert_examples_to_features(ReviewExample)
    """

    def __init__(self, input_ids, attention_mask, tag_ids, label_id, has_rationale, start_labels, end_labels, span_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.tag_ids = tag_ids
        self.label_id = label_id
        self.has_rationale = has_rationale
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.span_labels = span_labels
        
        

class DataProcessor(object):
    """Base class for data converters for rationale identification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of examples for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of examples for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """ Reads the data """
        lines = open(input_file).readlines()
        return lines

    
class MovieReviewsProcessor(DataProcessor):

    def __init__(self):
        self._tags = ['START', 'END', '0', '1']
        
        self._tag_map = {tag: i for i, tag in enumerate(self._tags)}

        self.fraction_rationales = 1.0

    def set_fraction_rationales(self, fraction_rationales):
        self.fraction_rationales = fraction_rationales

    def get_train_examples(self, data_dir):
            return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "dev.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return ["0", "1"]

    def get_tags(self):
        return self._tags

    def get_num_labels(self):
        return len(self.get_labels())

    def get_num_tags(self):
        return len(self._tags)

    def get_tag_map(self):
        return self._tag_map
    
    def get_start_tag_id(self):
        return self._tag_map['START']

    def get_stop_tag_id(self):
        return self._tag_map['END']

    def _create_examples(self, examples):
        # BOGUS method 
        return examples

    def _read_data_with_block(self, filename):

        content_lines = open(filename).readlines()
        tag_lines = open(filename + ".block").readlines()

        for idx in range(len(content_lines)):
            content_lines[idx] = content_lines[idx].replace("</POS>", "")
            content_lines[idx] = content_lines[idx].replace("<POS>", "")
            content_lines[idx] = content_lines[idx].replace("</NEG>", "")
            content_lines[idx] = content_lines[idx].replace("<NEG>", "")

        # remove certain rationales from the train set if needed
        if "train" in filename and self.fraction_rationales != 1.0:
            # print ("debug mode --- filtering out sentences")
            for idx in range(len(tag_lines)):
                # allow the input if it does *NOT* contain any rationale
                if "-1" in tag_lines[idx]:
                    continue
                # it contains a rationale 
                if random.random() > self.fraction_rationales:
                    # remove the rationale
                    tag_lines[idx] = " ".join(["-1" for _ in \
                        range(len(tag_lines[idx].strip().split()))])

        return [*zip(content_lines, tag_lines)]

class EsnliProcessor(DataProcessor):
  
    def __init__(self):
        self._tags = ['START', 'END', '0', '1', '2']
        
        self._tag_map = {tag: i for i, tag in enumerate(self._tags)}

        self.fraction_rationales = 1.0

    def set_fraction_rationales(self, fraction_rationales):
        self.fraction_rationales = fraction_rationales

    def get_train_examples(self, data_dir):
            return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "dev.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return ["0", "1", "2"]

    def get_tags(self):
        return self._tags

    def get_num_labels(self):
        return len(self.get_labels())

    def get_num_tags(self):
        return len(self._tags)

    def get_tag_map(self):
        return self._tag_map
    
    def get_start_tag_id(self):
        return self._tag_map['START']

    def get_stop_tag_id(self):
        return self._tag_map['END']

    def _create_examples(self, examples):
        # BOGUS method 
        return examples

    def _read_data_with_block(self, filename):

        content_lines = open(filename).readlines()
        tag_lines = open(filename + ".block").readlines()

        for idx in range(len(content_lines)):
            content_lines[idx] = content_lines[idx].replace("</POS>", "")
            content_lines[idx] = content_lines[idx].replace("<POS>", "")
            content_lines[idx] = content_lines[idx].replace("</NEG>", "")
            content_lines[idx] = content_lines[idx].replace("<NEG>", "")

        # remove certain rationales from the train set if needed
        if "train" in filename and self.fraction_rationales != 1.0:
            # print ("debug mode --- filtering out sentences")
            for idx in range(len(tag_lines)):
                # allow the input if it does *NOT* contain any rationale
                if "-1" in tag_lines[idx]:
                    continue
                # it contains a rationale 
                if random.random() > self.fraction_rationales:
                    # remove the rationale
                    tag_lines[idx] = " ".join(["-1" for _ in \
                        range(len(tag_lines[idx].strip().split()))])

        return [*zip(content_lines, tag_lines)]

class EvinfProcessor(DataProcessor):
  
    def __init__(self):
        self._tags = ['START', 'END', '0', '1', '2']
        
        self._tag_map = {tag: i for i, tag in enumerate(self._tags)}

        self.fraction_rationales = 1.0

    def set_fraction_rationales(self, fraction_rationales):
        self.fraction_rationales = fraction_rationales

    def get_train_examples(self, data_dir):
            return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "dev.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return ["0", "1", "2"]

    def get_tags(self):
        return self._tags

    def get_num_labels(self):
        return len(self.get_labels())

    def get_num_tags(self):
        return len(self._tags)

    def get_tag_map(self):
        return self._tag_map
    
    def get_start_tag_id(self):
        return self._tag_map['START']

    def get_stop_tag_id(self):
        return self._tag_map['END']

    def _create_examples(self, examples):
        # BOGUS method 
        return examples

    def _read_data_with_block(self, filename):

        content_lines = open(filename).readlines()
        tag_lines = open(filename + ".block").readlines()

        for idx in range(len(content_lines)):
            content_lines[idx] = content_lines[idx].replace("</POS>", "")
            content_lines[idx] = content_lines[idx].replace("<POS>", "")
            content_lines[idx] = content_lines[idx].replace("</NEG>", "")
            content_lines[idx] = content_lines[idx].replace("<NEG>", "")

        # remove certain rationales from the train set if needed
        if "train" in filename and self.fraction_rationales != 1.0:
            # print ("debug mode --- filtering out sentences")
            for idx in range(len(tag_lines)):
                # allow the input if it does *NOT* contain any rationale
                if "-1" in tag_lines[idx]:
                    continue
                # it contains a rationale 
                if random.random() > self.fraction_rationales:
                    # remove the rationale
                    tag_lines[idx] = " ".join(["-1" for _ in \
                        range(len(tag_lines[idx].strip().split()))])

        return [*zip(content_lines, tag_lines)]

class MultircProcessor(DataProcessor):
  
    def __init__(self):
        self._tags = ['START', 'END', '0', '1']
        
        self._tag_map = {tag: i for i, tag in enumerate(self._tags)}

        self.fraction_rationales = 1.0

    def set_fraction_rationales(self, fraction_rationales):
        self.fraction_rationales = fraction_rationales

    def get_train_examples(self, data_dir):
            return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "dev.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return ["0", "1"]

    def get_tags(self):
        return self._tags

    def get_num_labels(self):
        return len(self.get_labels())

    def get_num_tags(self):
        return len(self._tags)

    def get_tag_map(self):
        return self._tag_map
    
    def get_start_tag_id(self):
        return self._tag_map['START']

    def get_stop_tag_id(self):
        return self._tag_map['END']

    def _create_examples(self, examples):
        # BOGUS method 
        return examples

    def _read_data_with_block(self, filename):

        content_lines = open(filename).readlines()
        tag_lines = open(filename + ".block").readlines()

        for idx in range(len(content_lines)):
            content_lines[idx] = content_lines[idx].replace("</POS>", "")
            content_lines[idx] = content_lines[idx].replace("<POS>", "")
            content_lines[idx] = content_lines[idx].replace("</NEG>", "")
            content_lines[idx] = content_lines[idx].replace("<NEG>", "")

        # remove certain rationales from the train set if needed
        if "train" in filename and self.fraction_rationales != 1.0:
            # print ("debug mode --- filtering out sentences")
            for idx in range(len(tag_lines)):
                # allow the input if it does *NOT* contain any rationale
                if "-1" in tag_lines[idx]:
                    continue
                # it contains a rationale 
                if random.random() > self.fraction_rationales:
                    # remove the rationale
                    tag_lines[idx] = " ".join(["-1" for _ in \
                        range(len(tag_lines[idx].strip().split()))])

        return [*zip(content_lines, tag_lines)]

class PropagandaProcessor(DataProcessor):

    def __init__(self):
        self._tags = ['START', 'END', '0', '1']
        
        self._tag_map = {tag: i for i, tag in enumerate(self._tags)}
        self.fraction_rationales = 1.0

    def set_fraction_rationales(self, fraction_rationales):
        self.fraction_rationales = fraction_rationales

    def get_train_examples(self, data_dir):
            return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "dev.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return ["0", "1"]

    def get_tags(self):
        return self._tags

    def get_num_labels(self):
        return len(self.get_labels())

    def get_num_tags(self):
        return len(self._tags)

    def get_tag_map(self):
        return self._tag_map
    
    def get_start_tag_id(self):
        return self._tag_map['START']

    def get_stop_tag_id(self):
        return self._tag_map['END']

    def _create_examples(self, examples):
        # BOGUS method 
        return examples


    def _read_data_with_block(self, filename):

        content_lines = open(filename).readlines()
        tag_lines = open(filename + ".block").readlines()

        # remove certain rationales from the train set if needed
        if "train" in filename and self.fraction_rationales != 1.0:
            # print ("debug mode --- filtering out sentences")
            for idx in range(len(tag_lines)):
                # allow the input if it does *NOT* contain any rationale
                if "-1" in tag_lines[idx]:
                    continue
                # it contains a rationale 
                if random.random() > self.fraction_rationales:
                    # remove the rationale
                    tag_lines[idx] = " ".join(["-1" for _ in \
                        range(len(tag_lines[idx].strip().split()))])

        return [*zip(content_lines, tag_lines)]


class MultiRCProcessor(DataProcessor):

    def __init__(self):
        self._tags = ['START', 'END', '0', '1']
        # different tag for question, middle sep, and answer?
        
        self._tag_map = {tag: i for i, tag in enumerate(self._tags)}
        self.fraction_rationales = 1.0

    def set_fraction_rationales(self, fraction_rationales):
        self.fraction_rationales = fraction_rationales

    def get_train_examples(self, data_dir):
            return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "dev.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data_with_block(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return ["0", "1"]

    def get_tags(self):
        return self._tags

    def get_num_labels(self):
        return len(self.get_labels())

    def get_num_tags(self):
        return len(self._tags)

    def get_tag_map(self):
        return self._tag_map
    
    def get_start_tag_id(self):
        return self._tag_map['START']

    def get_stop_tag_id(self):
        return self._tag_map['END']

    def _create_examples(self, examples):
        # BOGUS method 
        return examples


    def _read_data_with_block(self, filename):

        content_lines = open(filename).readlines()
        tag_lines = open(filename + ".block").readlines()

        # remove certain rationales from the train set if needed
        if "train" in filename and self.fraction_rationales != 1.0:
            # print ("debug mode --- filtering out sentences")
            for idx in range(len(tag_lines)):
                # allow the input if it does *NOT* contain any rationale
                if "-1" in tag_lines[idx]:
                    continue
                # it contains a rationale 
                if random.random() > self.fraction_rationales:
                    # remove the rationale
                    tag_lines[idx] = " ".join(["-1" for _ in \
                        range(len(tag_lines[idx].strip().split()))])

        return [*zip(content_lines, tag_lines)]


def tags_to_positions(tags, has_rationale, max_len):
    seq_length = max_len # len(tags)
    start_labels, end_labels, span_labels = [0] * seq_length, [0] * seq_length, [[0]*seq_length]*seq_length
    span_labels = np.array(span_labels)
    if has_rationale:
        tlen = len(tags)
        start_positions = []
        end_positions = []
        for i in range(tlen):
            if (i-1 >= 0) and (tags[i] == 3 and tags[i-1] != 3):
                start_labels[i] = 1
                start_positions.append(i)
            if (i+1 <= tlen-1) and (tags[i+1] != 3 and tags[i] == 3):
                end_labels[i] = 1
                end_positions.append(i)
        # for i in range(seq_length):
        #     for j in range(seq_length):
        #         if start_labels[i] == 1 and end_labels[j] == 1:
        #             span_labels[i,j] = 1
        for start, end in zip(start_positions, end_positions):
            if start >= tlen or end >= tlen:
                continue
            span_labels[start, end] = 1

        #############################################################
        # idx = 0
        # while idx < tlen:
        #     row_id = -1
        #     col_id = -1
        #     if tags[idx] == 3:
        #         row_id = idx
        #         while (idx < tlen) and tags[idx] == 3:
        #             idx = idx + 1 
        #         if idx < tlen:
        #             col_id = idx - 1
        #         if row_id != -1 and col_id != -1:
        #             span_labels[row_id][col_id] = 1
        #     idx = idx + 1
        # # print(tags)
        ##############################################################
        sstart_sum = sum(start_labels)
        # print()
        # print(sum(end_labels))
        # print(sum(sum(span_labels)))
        # print()
        # if sstart_sum == 0:
        #     has_rationale = 0
        # print(start_labels)
        # print(end_labels)
        # print(span_labels)
        # for ind in range(tlen):
        #     if start_labels[ind] == 1:
        #         assert tags[ind] == 3
        #     if end_labels[ind] == 1:
        #         assert tags[ind] == 3
        #     for ind_j in range(tlen):
        #         if span_labels[ind][ind_j] == 1:
        #             assert tags[ind] == 3
        #             assert tags[ind_j] == 3
    return start_labels, end_labels, span_labels, has_rationale


def input_to_features(example, tokenizer, tag_map, max_seq_len):
    # news example is a tuple of content, tag 
    content, tags_input = example

    label = content.split("\t")[0]
    content = content.split("\t")[1]

    tokens = ['[CLS]']
    tags = [tag_map['START']]

    has_rationale = not ("-1" in tags_input)

    assert len(content.strip().split()) == len(tags_input.strip().split())

    for i, (word, tag) in enumerate(zip(content.strip().split(), tags_input.strip().split())):

        if word == "sep_token":
            tokens.append("[SEP]")
            tags.append(tag_map['0'])
            continue
        
        sub_words = tokenizer.tokenize(word)

        if not sub_words or len(sub_words) == 0:
            # can this even happen?? YES.... it does happen!!
            # print ("Ignoring the weird word: ", word)
            continue

        tokens.extend(sub_words)
        if tag != "-1":
            tags.extend([tag_map[tag] for _ in range(len(sub_words))])
        else:
            tags.extend([-1 for _ in range(len(sub_words))])


    tokens = tokens[:max_seq_len-1]
    tags = tags[:max_seq_len-1]

    tokens.append("[SEP]")
    tags.append(tag_map['END'])

    # print ('label = ', label)
    # print ('content = ', content)
    # print (tabulate(zip(tokens, tags)))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)    

    attention_mask = [1] * len(tokens)
    # if has_rationale:
    #     print(tags_input)
    start_labels, end_labels, span_labels, has_rationale = tags_to_positions(tags, has_rationale, max_seq_len)
    return InputFeatures(input_ids, attention_mask, tags, int(label), int(has_rationale), start_labels, end_labels, span_labels)



class DatasetWithRationales(data.Dataset):
    def __init__(self, examples, tokenizer, tag_map, max_seq_len, dataset="movie_reviews"):
        super(DatasetWithRationales, self).__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.tag_map = tag_map
        self.max_seq_len = max_seq_len
        self.dataset = dataset
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # if self.dataset == "movie_reviews": # modified delete
        if self.dataset in ['movie_reviews', 'esnli', 'evinf', 'multirc']:  # modified add
            features = input_to_features(self.examples[idx], self.tokenizer, 
                        self.tag_map, self.max_seq_len)
        elif self.dataset == "propaganda":
            features = input_to_features(self.examples[idx], self.tokenizer, 
                        self.tag_map, self.max_seq_len)
        elif self.dataset == "multi_rc":
            features = input_to_features(self.examples[idx], self.tokenizer, 
                        self.tag_map, self.max_seq_len)
        else:
            raise Exception("No dataset selected....")

        
        return features.input_ids, features.attention_mask, \
            features.label_id, features.tag_ids, features.has_rationale, features.start_labels, features.end_labels, features.span_labels
        
    @classmethod
    def pad(cls, batch):

        float_type = torch.FloatTensor
        long_type = torch.LongTensor
        bool_type = torch.bool

        is_cuda = torch.cuda.is_available()

        if is_cuda:
            float_type = torch.cuda.FloatTensor 
            long_type = torch.cuda.LongTensor


        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] 
        f_single = lambda x: [sample[x] for sample in batch]
        
        f_two_dims = lambda x, seqlen: [np.array(sample[x])[:seqlen, :seqlen] for sample in batch] 
        f_one_dims = lambda x, seqlen: [np.array(sample[x])[:seqlen] for sample in batch]
        # 0: X for padding

        # input_ids, attention_mask, label_ids, tag_ids, has_rationale = batch

        input_ids_list = torch.Tensor(f(0, maxlen)).type(long_type)
        attention_mask_list = torch.Tensor(f(1, maxlen)).type(long_type)

        label_ids_list = torch.Tensor(f_single(2)).type(long_type) 
        tag_ids_list = torch.Tensor(f(3, maxlen)).type(long_type)
        rationale_list = torch.Tensor(f_single(4)).type(bool_type) 
        start_labels_list = torch.Tensor(f_one_dims(5, maxlen)).type(long_type)
        end_labels_list = torch.Tensor(f_one_dims(6, maxlen)).type(long_type)
        span_labels_list = torch.Tensor(f_two_dims(7, maxlen)).type(long_type)
        return input_ids_list, attention_mask_list, label_ids_list, tag_ids_list, rationale_list, start_labels_list, end_labels_list, span_labels_list

if __name__ == "__main__":
    start_labels, end_labels, span_labels, _ = tags_to_positions(tags=[0, 2, 3, 2, 3, 2, 3, 1], has_rationale=1, max_len=10)
    print(start_labels)
    print()
    print(end_labels)
    print()
    print(span_labels)
    print()