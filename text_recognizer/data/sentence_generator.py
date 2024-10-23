from text_recognizer.data.base_data_module import BaseDataModule
from typing import List

import itertools
import nltk
import numpy as np
import re
import string

NLTK_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "nltk"

def load_nltk_brown_corpse() -> List[List[str]]:
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    
    try:
        nltk.corpus.brown.sents()
    except LookupError:
        NLTK_DATA_DIRNAME.mkdir(parents = True, exist_ok = True)
        nltk.download(info_or_id = "brown", download_dir = NLTK_DATA_DIRNAME)
        
    return nltk.corpus.brown.sents()

def brown_text() -> str:
    sents = load_nltk_brown_corpse()
    text = " ".join(itertools.chain().from_iterable(sents))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(" +", " ", text)
    return text

class SentenceGenerator:
    def __init__(self, max_length = None):
        self.text= brown_text()
        self.word_start_idxs = [0] + [_.start(0) + 1 for _ in re.finditer(" ", self.text)]
        self.max_length = max_length
        
    def generate(self, max_length = None):
        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError("max_length must be provided to this function or when instantiating this object.")
        
        sampled_text, num_tries = None, 0
        
        while (not sampled_text) and (num_tries <= 10):
            first_idx = np.random.randint(0, len(self.word_start_idxs))
            start_idx = self.word_start_idxs[first_idx]
            
            end_idx_candidates = self._get_end_idx_candidates(first_idx, start_idx, max_length)
            
            if len(end_idx_candidates) == 0:
                num_tries += 1
                continue
            else:
                end_idx = np.random.choice(end_idx_candidates)
                sampled_text = self.text[start_idx:end_idx].strip()
                
        if sampled_text is not None:
            return sampled_text
        else:
            raise RuntimeError("Could not generate a valid string.")
            
    def _get_end_idx_candidates(self, first_idx, start_idx, max_length):
        end_idx_candidates = []
        
        for idx in range(first_idx + 1, len(self.word_start_idxs)):
            if self.word_start_idxs[idx] - start_idx > max_length:
                break
            end_idx_candidates.append(self.word_start_idxs[idx])
            
        return end_idx_candidates