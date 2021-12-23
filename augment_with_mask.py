# import the CheckListAugmenter
from textattack.augmentation import (
    WordNetAugmenter,
    EmbeddingAugmenter,
    CharSwapAugmenter,
    EasyDataAugmenter,
    CheckListAugmenter,
    DeletionAugmenter,
    CLAREAugmenter,
)
# from textattack.commands.attack 
import numpy as np
# # Alter default values if desired
# import nltk
# from textattack.constraints import PreTransformationConstraint
# from textattack.shared.validators import transformation_consists_of_word_swaps


# class MaskedModification(PreTransformationConstraint):
#     """A constraint disallowing the modification of stopwords."""

#     def __init__(self, mask=None):
#         self.mask = mask

#     def _get_modifiable_indices(self, current_text):
#         """Returns the word indices in ``current_text`` which are able to be
#         modified."""
#         non_stopword_indices = set()
#         assert len(self.mask) == len(current_text.words)#, print(len(self.mask), " ", len(current_text.words))
#         for i, word in enumerate(current_text.words):
#             if int(self.mask[i]) == 0:
#                 non_stopword_indices.add(i)
#         return non_stopword_indices

#     def check_compatibility(self, transformation):
#         """The stopword constraint only is concerned with word swaps since
#         paraphrasing phrases containing stopwords is OK.

#         Args:
#             transformation: The ``Transformation`` to check compatibility with.
#         """
#         return transformation_consists_of_word_swaps(transformation)


# class MaskedCheckListAugmenter(Augmenter):
#     def __init__(self, myconstraints=None, **kwargs):
#         from textattack.transformations import (
#             CompositeTransformation,
#             WordSwapChangeLocation,
#             WordSwapChangeName,
#             WordSwapChangeNumber,
#             WordSwapContract,
#             WordSwapExtend,
#         )

#         transformation = CompositeTransformation(
#             [
#                 WordSwapChangeNumber(),
#                 WordSwapChangeLocation(),
#                 WordSwapChangeName(),
#                 WordSwapExtend(),
#                 WordSwapContract(),
#             ]
#         )

#         constraints = [MaskedModification(myconstraints)]

#         super().__init__(transformation, constraints=constraints, **kwargs)

def masked_augment(sentence, mask, augmenter):
    subsents = []
    masked_sentences = []
    words = sentence.strip().split()
    tlen = len(words)
    assert len(words) == len(mask)
    idx = 0
    row_id = -1
    col_id = -1
    mask.append(0)
    tlen += 1
    while idx < tlen:
       
        row_id = -1
        col_id = -1
        if int(mask[idx]) == 0:
            row_id = idx
            while (idx < tlen) and int(mask[idx]) == 0:
                idx = idx + 1 
            if idx < tlen:
                col_id = idx - 1
            if row_id != -1 and col_id != -1:
                subsents.append((" ".join(words[row_id:col_id+1]), row_id, col_id+1))
        idx = idx + 1
    if row_id != tlen - 1:
        subsents.append((" ".join(words[row_id:tlen-1]), row_id, tlen-1))
    # print(subsents)
    mask.pop(-1)
    idx = 0
    new_words = []
    new_block = []
    for subsen in subsents:
        new_text = augmenter.augment(subsen[0])[0].split()
        c_new_text = " ".join(new_text)
        # print(subsen[0])
        # if c_new_text != subsen[0]:
        #     print(augmenter.augment(subsen[0]))
        #     print(subsen[0])
        #     print()
        
        # new_text = subsen[0]
        # new_text += " </sep>"
        # new_text = "<sep> " + new_text
        new_block.extend(['1'] * len(words[idx:subsen[1]]))
        new_words.extend(words[idx:subsen[1]])
        new_block.extend(['0'] * len(new_text))
        new_words.extend(new_text)
        idx = subsen[2]
    if len(new_block) == 0:
        new_block = mask
        new_words = words
    return new_words, new_block


if __name__ == "__main__":
    # ['I would love to go to Japan but the tickets are 792 dollars', "I'd love to go to British Virgin Islands but the tickets are 814 dollars", "I'd love to go to Channel Islands but the tickets are 489 dollars", "I'd love to go to Guyana but the tickets are 154 dollars", "I'd love to go to West Bank and Gaza but the tickets are 500 dollars"]
    s = "\"The Saint Takes Over\" stars George Sanders as Simon Templar, aka \"The Saint\" in this 1940 entry into the series. It also stars Wendy Barrie, Jonathan Hale and Paul Guilfoyle ...It is very enjoyable."
    # c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c = len(s.split()) * [0]
    # c = []
    # Augment
    augmenter = CheckListAugmenter(pct_words_to_swap=0.2)#, transformations_per_example=10)
    import time 
    new_text = augmenter.augment("it is a enigma how the film wo be releases in this condition.")
    print(new_text)
    # times = []
    # for i in range(1):
    #     start_time = time.time()
    #     masked_sentences, _ = masked_augment(s, c, augmenter=augmenter)
    #     times.append(time.time() - start_time)
    #     print(" ".join(masked_sentences))
    #     print("time: {}".format(times[-1]))
    # print("mean: ", np.mean(times))
    
    '''
    
    for split_name in ["train"]:#, "dev", "test"]:
        # /home/ldf/evidence-extraction/datasets/v0_data/{}.txt
        data_file = "/home/ldf/evidence-extraction/datasets/movie_reviews_only_rats/{}.txt".format(split_name)
        with open(data_file) as f:
            data = f.readlines()
            text = [x.strip() for x in data] 
        with open(data_file+".block") as f:
            data = f.readlines()
            block = [x.strip() for x in data]
        assert len(text) == len(block)
        add = 0
        new_text = []
        new_block = []
        for i, (t, b) in enumerate(zip(text, block)):
            if ("-1" in b):
                continue
            if i != 15:
                continue
            la, te = t.split("\t")[0:2]
            
            te = te.replace("</POS>", "")
            te = te.replace("<POS>", "")
            te = te.replace("</NEG>", "")
            te = te.replace("<NEG>", "")
            # print(len(te.split()))多线程
            # print(len(b))
            b = b.strip().split()
            # try:
            masked_words, masked_block= masked_augment(te, b, augmenter)
        
            new_text.append("{}\t{}".format(la, " ".join(masked_words)))
            assert len(masked_words) == len(masked_block)
            new_block.append(" ".join(masked_block))
            add += 1
        
            exit(0)
        new_text = text + new_text
        new_block = block + new_block
        # with open("/home/ldf/evidence-extraction/datasets/mulirc_adv/{}.txt".format(split_name), "w") as f:
        #     f.write("\n".join(new_text))
        # with open("/home/ldf/evidence-extraction/datasets/mulirc_adv/{}.txt.block".format(split_name), "w") as f:
        #     f.write("\n".join(new_block))

    '''