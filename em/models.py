from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


def nonzero(a): 
    # return np.clip(a, a_min=1e-12, a_max=None)
    return a + 1e-12

class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        res = []
        # theta = self.translation_probs # (vocab_src, vocab_trg)
        for e in parallel_corpus:
            s, t = e.source_tokens, e.target_tokens
            n, m = len(t), len(s)
            q = np.zeros([n, m])
            if n == 0 or m == 0:
                res.append(q)
                continue
            p = self.translation_probs[np.ix_(s, t)]  # (src_len, trg_len)
            d = p.sum(axis=0, keepdims=True)
            q = p / nonzero(d)
            res.append(q)
        return res

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """

        elbo = 0.0
        for k, e in enumerate(parallel_corpus):
            s, t = e.source_tokens, e.target_tokens
            q = posteriors[k]
            n, m = len(t), len(s)
            if n == 0 or m == 0: 
                continue
            
            # alignment prior
            prior = -n * np.log(m)

            p = self.translation_probs[np.ix_(s, t)]  # (src_len, trg_len)
            elbo += prior + np.sum(q * np.log(nonzero(p / nonzero(q))))
        return elbo

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        # Q = posteriors
        self.translation_probs *= 0
        # calculate lambdas
        for k, e in enumerate(parallel_corpus):
            # q = Q[k]
            s = e.source_tokens
            t = e.target_tokens
            n, m = len(t), len(s)

            if n == 0 or m == 0:
                continue

            # vectorized update with broadcasting
            np.add.at(self.translation_probs, np.ix_(s, t), posteriors[k])

        self.translation_probs /= np.sum(self.translation_probs, axis=1).reshape(-1, 1) # deviding each row by d
        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        res = []
        theta = self.translation_probs
        for e in sentences:
            s, t = e.source_tokens, e.target_tokens
            n, m = len(t), len(s)
            if n == 0 or m == 0:
                res.append([])
                continue
            # greedy decoding
            p = theta[np.ix_(s, t)]
            a = np.argmax(p, axis=0) + 1 # indexes of p(t_i | S_a)
            res.append(list(zip(a, np.arange(1, m + 1))))
        return res

class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, n: int, m: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        key = (n, m)
        if key not in self.alignment_probs:
            # uniform
            self.alignment_probs[key] = np.full((n, m), 1.0 / n, dtype=np.float32)
        return self.alignment_probs[key]

    def _e_step(self, parallel_corpus):
        res = []
        # theta = self.translation_probs # (vocab_src, vocab_trg)
        for e in parallel_corpus:
            s, t = e.source_tokens, e.target_tokens
            n, m = len(t), len(s)
            q = np.zeros([n, m])
            if n == 0 or m == 0:
                res.append(q)
                continue
            phi = self._get_probs_for_lengths(m, n)
            
            p = self.translation_probs[np.ix_(s, t)] * phi   # (n, m)
            # print(phi.shape, m, n, p.shape)
            
            d = p.sum(axis=0, keepdims=True)
            q = p / nonzero(d)
            res.append(q)
        return res

    def _compute_elbo(self, parallel_corpus, posteriors):
        elbo = 0.0
        for k, e in enumerate(parallel_corpus):
            s, t = e.source_tokens, e.target_tokens
            q = posteriors[k]
            n, m = len(t), len(s)
            if n == 0 or m == 0: 
                continue

            phi = self._get_probs_for_lengths(m, n)

            p = self.translation_probs[np.ix_(s, t)]  # (src_len, trg_len)
            elbo += np.sum(q * (np.log(nonzero(p)) + np.log(nonzero(phi)) - np.log(nonzero(q))))
        return elbo

    def _m_step(self, parallel_corpus, posteriors):
        Q = posteriors
        self.translation_probs *= 0
        self.alignment_probs.clear()

        # group posteriors by (src_len, trg_len)
        from collections import defaultdict
        length_groups = defaultdict(list)

        for k, e in enumerate(parallel_corpus):
            q = Q[k]
            s = e.source_tokens
            t = e.target_tokens
            n, m = len(t), len(s)

            length_groups[(m, n)].append(q)

            if n == 0 or m == 0:
                continue

            # vectorized update with broadcasting
            np.add.at(self.translation_probs, np.ix_(s, t), q)
        
        # update alignment_probs (phi) for each (src_len, tgt_len)
        for (m, n), qs in length_groups.items():
            total = np.zeros((n, m))
            for q in qs:
                # print(total.shape, q.shape)
                # print(total.shape, q.shape)

                total += q.T

            d = total.sum(axis=1, keepdims=True)

            
            phi_new = total / nonzero(d)
            self.alignment_probs[(m, n)] = phi_new.T

        self.translation_probs /= np.sum(self.translation_probs, axis=1).reshape(-1, 1)
        return self._compute_elbo(parallel_corpus, posteriors)

    def align(self, sentences):
        res = []
        theta = self.translation_probs
        for e in sentences:
            s, t = e.source_tokens, e.target_tokens
            n, m = len(t), len(s)
            if n == 0 or m == 0:
                res.append([])
                continue
            # greedy decoding
            phi = self._get_probs_for_lengths(m, n)
            p = theta[np.ix_(s, t)] * phi
            a = np.argmax(p, axis=0) + 1 # indexes of p(t_i | S_a)
            res.append(list(zip(a, np.arange(1, m + 1))))
        return res