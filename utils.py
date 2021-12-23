from transformers import BertTokenizer, BertForSequenceClassification
import os



def load_tokenizer(path="../transformers-local/tmp/propaganda/"):
    """Load and return an instance of Bert Tokenizer from the given path
    
    Parameters
    ----------
    path : str, optional
        Path to the tokenization directory, by default "../transformers-local/tmp/propaganda/"
    
    Returns
    -------
    BertTokenizer
        Returns a tokenizer
    """
    print ("loading tokenizer...")    
    tokenizer = BertTokenizer.from_pretrained(path)
    print ("done loading tokenizer.")    
    return tokenizer


def load_model(path="../transformers-local/tmp/propaganda/"):
    """Load and return an instance of BertForSequenceClassification model from the given path
    
    Parameters
    ----------
    path : str, optional
        Path to the tokenization directory, by default "../transformers-local/tmp/propaganda/"
    
    Returns
    -------
    BertForSequenceClassification 
        Returns a bert-based sequence classification model 
    """
    print ("loading the model ...")
    model = BertForSequenceClassification.from_pretrained(path, output_attentions=False)
    print ("Done loading the model...")

    return model


def tokenize(tokenizer, line, words_only=True):
    """Tokenize a line into byte pairs or words.
    
    Parameters
    ----------
    tokenizer : BertTokenizer
        An instance of the tokenizer
    line : str
        Input line
    words_only : bool, optional
        Whether we want words or byte pairs, by default True (i.e words)
    
    Returns
    -------
    list
        Returns a list of words or byte-pair tokens.
    """
    bp_tokens = tokenizer.tokenize(line)
    if words_only:
        words = club_byte_pairs(bp_tokens)
        return words
    return bp_tokens
    

def club_byte_pairs_and_scores(bp_tokens, scores):
    """ Helper function to collate the broken words into byte-pairs.

    The function also combines importance scores corresponding to the clubbed byte-pair tokens. 
    The broken words can be identified with "##". For instance, `frustrating' might be broken
    down to `frustrat` and `##ing`.
    
    Parameters
    ----------
    bp_tokens : list
        List of byte pair tokens, each byte pair token is a string.
    scores : list
        List of 1-d tensor values containing importance scores
    
    Returns
    -------
    list, list
        Returns the clubbed words and scores, respectively.
    """
    in_middle = False
    updated_scores = []
    updated_words = []
    current_idx = 0
    for i, bp_token in enumerate(bp_tokens):
        if bp_token[:2] == "##":
            updated_words[current_idx-1] += bp_token[2:]
            updated_scores[current_idx-1] += scores[i].item()
        else:
            updated_scores.append(scores[i].item())
            updated_words.append(bp_token)
            current_idx += 1

    return updated_words, updated_scores


def club_byte_pairs(bp_tokens):
    """ Helper function to collate the broken words into byte-pairs.

    The broken words can be identified with "##". For instance, `frustrating' might be broken
    down to `frustrat` and `##ing`.
    
    Parameters
    ----------
    bp_tokens : list
        List of byte pair tokens, each byte pair token is a string.
    
    Returns
    -------
    list
        Returns the clubbed words.
    """
    in_middle = False
    updated_words = []
    current_idx = 0
    for i, bp_token in enumerate(bp_tokens):
        if bp_token[:2] == "##":
            updated_words[current_idx-1] += bp_token[2:]
        else:
            updated_words.append(bp_token)
            current_idx += 1

    return updated_words


def ensure_dir(file_path):
    """Creates a directory if it does not exist 
    
    Parameters
    ----------
    file_path : str
        Path of the file
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


def separate_tags_from_lines(lines):
    """ Separates the tags and content from the input lines.
    
    Parameters
    ----------
    lines : list[str]
        List of input lines. 
    
    Returns
    -------
    list[str], list[str]
        Returns the respective lists of separated tags, and input content.

    """
    tags, articles = [], []
    for line in lines:
        tag, content = line.split("\t")[0], line.split("\t")[1]
        tags.append(tag)
        articles.append(content)
    return tags, articles