from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]

    def __repr__(self):
        return f"src:\t{' '.join(self.source)}\ntrg:\t{' '.join(self.target)}\n"

@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray

    def __repr__(self):
        return f"src:\t{', '.join(map(str, self.source_tokens))}\ntrg:\t{', '.join(map(str, self.target_tokens))}\n"


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]
    def __repr__(self):
        return f'sure  :\t{list(map(lambda x: (int(x[0]), int(x[1])), self.sure))}\npossible:\t{self.possible}'



def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    
    f = open(filename, 'r', encoding='utf-8')
    content = f.read()
    f.close()

    content = content.replace('&', '&amp;')

    root = ET.fromstring(content)

    english = list(map(lambda x: x.text.split(), root.findall('s/english')))
    czech = list(map(lambda x: x.text.split(), root.findall('s/czech')))
    assert len(english) == len(czech)

    sure = list(map(lambda x: x.text, root.findall('s/sure')))
    possible = list(map(lambda x: x.text, root.findall('s/possible')))

    def process(s):
        if s == None: return []
        l = s.split()
        l = [x.split('-') for x in l]
        l = [(int(x[0]), int(x[1])) for x in l]
        return l

    sure = list(map(process, sure))
    possible = list(map(process, possible))

    a = list(map(lambda x: SentencePair(*x), zip(english, czech)))
    b = list(map(lambda x: LabeledAlignment(*x), zip(sure, possible)))
    return a, b


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    src_cnt = Counter()
    trg_cnt = Counter()
    for p in sentence_pairs:
        src_cnt.update(p.source)
        trg_cnt.update(p.target)
    src, trg = {}, {}



    # tokens_src = sorted((src_cnt.keys()), key=src_cnt.get, reverse=True)
    # tokens_trg = sorted((trg_cnt.keys()), key=trg_cnt.get, reverse=True)


    if freq_cutoff is not None:
        tokens_src = map(lambda x:x[0], src_cnt.most_common(freq_cutoff))
        tokens_trg = map(lambda x:x[0],trg_cnt.most_common(freq_cutoff))
    else:
        tokens_src = src_cnt.keys()
        tokens_trg = trg_cnt.keys()


    for i, t in enumerate(tokens_src):
        src[t] = i
    for i, t in enumerate(tokens_trg):
        trg[t] = i
    return src, trg


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    res = []
    for s in sentence_pairs:
        tokens_src = list(filter(lambda x: x != None, list(map(source_dict.get, s.source))))
        tokens_trg = list(filter(lambda x: x != None, list(map(target_dict.get, s.target))))
        if len(tokens_src) == 0 or len(tokens_trg) == 0: continue
        res.append(TokenizedSentencePair(np.array(tokens_src), np.array(tokens_trg)))
    return res