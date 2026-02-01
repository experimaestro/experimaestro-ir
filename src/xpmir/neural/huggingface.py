from typing import List, Sequence, Tuple, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from experimaestro import Param
from xpm_torch.learner import TrainerContext
from datamaestro_text.data.ir import TextItem
from xpmir.text import TokenizedTexts

from xpmir.letor.records import BaseRecords
from xpmir.rankers import AbstractModuleScorer


class HFCrossScorer(AbstractModuleScorer):
    """Load a cross scorer model from the huggingface"""

    hf_id: Param[str]
    """the id for the huggingface model"""

    max_length: Param[Optional[int]] = None
    """the max length for the transformer model"""

    # Per-side maxima: queries and documents
    max_query_length: Param[Optional[int]] = 32
    """maximum number of tokens for the query side"""

    max_doc_length: Param[Optional[int]] = 256
    """maximum number of tokens for the document side"""

    def __post_init__(self):
        self.config = AutoConfig.from_pretrained(self.hf_id)

        #ensure that num_labels is one for a Cross-encoder
        if hasattr(self.config, "num_labels"):
            self.config.num_labels = 1
        else:
            self.logger.warning("no 'num_labels param found in config, check that classifier outputs one label")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_id, config=self.config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)

    def batch_tokenize(
        self,
        input_records: BaseRecords,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
        """Transform the text to tokens by using the tokenizer"""
        # determine per-side token limits (instance params take precedence)
        q_max = self.max_query_length if getattr(self, "max_query_length", None) is not None else None
        d_max = self.max_doc_length if getattr(self, "max_doc_length", None) is not None else None

        # Batch-truncate queries and documents separately to avoid N tokenizer calls.
        def _batch_truncate(texts: Sequence[str], max_tokens: Optional[int]) -> List[str]:
            """Truncate a list of texts to at most `max_tokens` tokens each and decode back to strings.

            If `max_tokens` is None, returns the original texts as a list.
            """
            if max_tokens is None:
                return list(texts)

            enc = self.tokenizer(
                list(texts),
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids = enc["input_ids"]
            return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        queries = [q[TextItem].text for q in input_records.queries]
        docs = [d[TextItem].text for d in input_records.documents]

        truncated_queries = _batch_truncate(queries, q_max)
        truncated_docs = _batch_truncate(docs, d_max)

        texts: List[Tuple[str, str]] = list(zip(truncated_queries, truncated_docs))

        # compute combined max length (respect model maximum)
        combined_limit = self.tokenizer.model_max_length
        if maxlen is not None:
            combined_limit = min(maxlen, combined_limit)
        if q_max is not None and d_max is not None:
            combined_limit = min(combined_limit, q_max + d_max)

        r = self.tokenizer(
            texts,
            max_length=combined_limit,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=mask,
        )
        return TokenizedTexts(
            None,
            r["input_ids"].to(self.device),
            r["length"],
            r.get("attention_mask", None),
            r.get("token_type_ids", None),  # if r["token_type_ids"] else None
        )

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):

        tokenized = self.batch_tokenize(inputs, maxlen=self.max_length, mask=True)
        # strange that some existing models on the huggingface don't use the token_type
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            result = self.model(
                tokenized.ids,
                token_type_ids=tokenized.token_type_ids.to(self.device),
                attention_mask=tokenized.mask.to(self.device),
            ).logits  # Tensor[float] of length records size
        return result

    def distribute_models(self, update):
        self.model = update(self.model)
