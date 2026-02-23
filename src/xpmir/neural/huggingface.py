from typing import List, Sequence, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from experimaestro import Param
from xpm_torch.learner import TrainerContext
from xpmir.text import TokenizedTexts

from xpmir.letor.records import BaseRecords
from xpmir.rankers import AbstractModuleScorer
from xpm_torch.utils import to_device


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

        assert self.max_doc_length > 0, "max_doc_length is not set"
        assert self.max_query_length > 0, "max_query_length is not set"

        self.config = AutoConfig.from_pretrained(self.hf_id)

        if self.max_length is None:
            original_max = self.config.max_position_embeddings
            self.logger.info(
                f"No max_length specified for {self.hf_id}, using model's original max_position_embeddings: {original_max}"
            )
            self.max_length = original_max
        else:
            if self.max_length > self.config.max_position_embeddings:
                self.logger.warning(
                    f"Specified max_length {self.max_length} exceeds model's max_position_embeddings {self.config.max_position_embeddings}. Capping to model's maximum."
                )
                self.max_length = self.config.max_position_embeddings

        # ensure that num_labels is one for a Cross-encoder
        if hasattr(self.config, "num_labels"):
            self.config.num_labels = 1
        else:
            self.logger.warning(
                "no 'num_labels param found in config, check that classifier outputs one label"
            )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_id,
            config=self.config,
            dtype=torch.float32,
        )

        if self.hf_id == "microsoft/MiniLM-L12-H384-uncased":
            self.logger.warning(
                "Enforcing lower_case to True for microsoft/MiniLM-L12-H384-uncased"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_id, do_lower_case=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)

    def batch_tokenize(
        self,
        input_records: BaseRecords,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
        """Transform the text to tokens by using the tokenizer"""
        # determine per-side token limits (instance params take precedence)
        q_max = (
            self.max_query_length
            if getattr(self, "max_query_length", None) is not None
            else None
        )
        d_max = (
            self.max_doc_length
            if getattr(self, "max_doc_length", None) is not None
            else None
        )

        # Batch-truncate queries and documents separately to avoid N tokenizer calls.
        def _batch_truncate(
            texts: Sequence[str], max_tokens: Optional[int]
        ) -> List[str]:
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
            return self.tokenizer.batch_decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        queries = [q["text_item"].text for q in input_records.queries]
        docs = [d["text_item"].text for d in input_records.documents]

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

    def batch_tokenize_v2(
        self,
        input_records: BaseRecords,
        maxlen: Optional[int] = None,
        mask: bool = False,
    ) -> TokenizedTexts:
        """Tokenize (query, document) pairs with per-side length limits.

        Compatible with fast tokenizers (TokenizersBackend) 
        Special tokens are inserted manually from the tokenizer's vocabulary.
        """
        q_max: Optional[int] = getattr(self, "max_query_length", None)
        d_max: Optional[int] = getattr(self, "max_doc_length", None)

        queries = [q["text_item"].text for q in input_records.queries]
        docs    = [d["text_item"].text for d in input_records.documents]

        # ------------------------------------------------------------------ #
        # Special token IDs (BERT-style: [CLS] q [SEP] d [SEP])              #
        # ------------------------------------------------------------------ #
        tok = self.tokenizer
        cls_id = tok.cls_token_id   # [CLS]
        sep_id = tok.sep_token_id   # [SEP]
        pad_id = tok.pad_token_id
        num_special = 3  # [CLS] + [SEP] + [SEP]

        # Verify the tokenizer has the tokens we expect
        assert cls_id is not None and sep_id is not None, (
            "Tokenizer must define cls_token and sep_token for pair encoding."
        )


        # Combined sequence length cap                                       #
        combined_limit = tok.model_max_length  # 8192 for ettin
        if maxlen is not None:
            combined_limit = min(maxlen, combined_limit)
        if q_max is not None and d_max is not None:
            combined_limit = min(combined_limit, q_max + d_max + num_special)

        content_limit = combined_limit - num_special  # tokens available for text

        def _encode(texts: List[str], max_tokens: Optional[int]) -> List[torch.Tensor]:
            enc = tok(
                texts,
                add_special_tokens=False,
                truncation=max_tokens is not None,
                max_length=max_tokens or tok.model_max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_tensors=None,          # keep as List[List[int]] — lengths differ
            )
            # Convert each sample to a 1-D tensor individually (lengths vary)
            return [torch.tensor(ids, dtype=torch.long) for ids in enc["input_ids"]]

        query_tensors = _encode(queries, q_max)
        doc_tensors   = _encode(docs,   d_max)

        
        # Assemble [CLS] q [SEP] d [SEP], respecting combined_limit
        cls = torch.tensor([cls_id], dtype=torch.long)
        sep = torch.tensor([sep_id], dtype=torch.long)

        sequences: List[torch.Tensor] = []
        lengths:   List[int]          = []

        for q_ids, d_ids in zip(query_tensors, doc_tensors):
            # Trim doc if combined still overflows
            overflow = (q_ids.size(0) + d_ids.size(0)) - content_limit
            if overflow > 0:
                d_ids = d_ids[:max(0, d_ids.size(0) - overflow)]

            seq = torch.cat([cls, q_ids, sep, d_ids, sep]) 
            sequences.append(seq)
            lengths.append(seq.size(0))

        # Pad to max length in batch using F.pad (no Python list building)
        max_len = max(lengths)
        input_ids = torch.stack([
            F.pad(seq, (0, max_len - seq.size(0)), value=pad_id)
            for seq in sequences
        ])  # (B, max_len)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        attention_mask = (
            torch.arange(max_len).unsqueeze(0) < lengths_tensor.unsqueeze(1)
        ).long() if mask else None  # (B, max_len)

        # token_type_ids: 0 for [CLS]+q+[SEP], 1 for d+[SEP]
        token_type_ids = torch.stack([
            F.pad(
                torch.ones(len(d) + 1, dtype=torch.long),   # doc segment
                (max_len - len(d) - 1, 0),                   # pad left with 0s (query side)
                value=0,
            )
            for d, length in zip(doc_tensors, lengths)
            # Flip: build from right so query side is naturally 0
        ])
        
        q_lengths = torch.tensor([q.size(0) + 2 for q in query_tensors])  # +[CLS]+[SEP]
        token_type_ids = (
            torch.arange(max_len).unsqueeze(0) >= q_lengths.unsqueeze(1)
        ).long()  # 0 for query side, 1 for doc side

        return TokenizedTexts(
            None,
            to_device(input_ids, self.device),
            torch.tensor(lengths),          # pre-padding per-sample lengths
            to_device(attention_mask,self.device),
            to_device(token_type_ids,self.device),
        )

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        tokenized = self.batch_tokenize_v2(inputs, maxlen=self.max_length, mask=True)
        # strange that some existing models on the huggingface don't use the token_type
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            result = self.model(
                tokenized.ids,
                token_type_ids=to_device(tokenized.token_type_ids, self.device),
                attention_mask=to_device(tokenized.mask, self.device),
            ).logits  # Tensor[float] of length records size
        return result
