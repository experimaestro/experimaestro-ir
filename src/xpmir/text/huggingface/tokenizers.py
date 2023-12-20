from typing import List
from experimaestro import Param
from xpmir.text.tokenizers import TokenizedTexts, ListTokenizer
from .base import HFTokenizer


class HFListTokenizer(ListTokenizer):
    """Process list of texts by separating them by a separator token"""

    tokenizer: Param[HFTokenizer]
    """The HF tokenizer"""

    def tokenize(self, text_lists: List[List[str]]) -> TokenizedTexts:
        assert self.tokenizer.sep_token is not None
        sep = f" {self.tokenizer.cls_token} "

        self.tokenizer.batch_tokenize([sep.join(text_list) for text_list in text_lists])
