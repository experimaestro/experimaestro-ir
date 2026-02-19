from typing import List, Sequence, Tuple, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from experimaestro import Param
from xpm_torch.learner import TrainerContext
from datamaestro_text.data.ir import TextItem
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
        self.config = AutoConfig.from_pretrained(self.hf_id)

        if self.max_length is None:
            original_max = self.config.max_position_embeddings
            self.logger.info(f"No max_length specified for {self.hf_id}, using model's original max_position_embeddings: {original_max}")
            self.max_length = original_max
        else:            
            if self.max_length > self.config.max_position_embeddings:
                self.logger.warning(f"Specified max_length {self.max_length} exceeds model's max_position_embeddings {self.config.max_position_embeddings}. Capping to model's maximum.")
                self.max_length = self.config.max_position_embeddings
        
        #ensure that num_labels is one for a Cross-encoder
        if hasattr(self.config, "num_labels"):
            self.config.num_labels = 1
        else:
            self.logger.warning("no 'num_labels param found in config, check that classifier outputs one label")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_id, config=self.config, dtype=torch.float32,
        )

        if self.hf_id == "microsoft/MiniLM-L12-H384-uncased":
            self.logger.warning("Enforcing lower_case to True for microsoft/MiniLM-L12-H384-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, do_lower_case=True)
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
    
    def batch_tokenize_v2(
        self,
        input_records: BaseRecords,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
        """Transform the text to tokens by using the tokenizer"""
        # determine per-side token limits (instance params take precedence)
        q_max = self.max_query_length if getattr(self, "max_query_length", None) is not None else None
        d_max = self.max_doc_length if getattr(self, "max_doc_length", None) is not None else None

        queries = [q[TextItem].text for q in input_records.queries]
        docs = [d[TextItem].text for d in input_records.documents]

        # compute combined max length (respect model maximum)
        combined_limit = self.tokenizer.model_max_length
        if maxlen is not None:
            combined_limit = min(maxlen, combined_limit)
        if q_max is not None and d_max is not None:
            combined_limit = min(combined_limit, q_max + d_max)

        # Encode without special tokens and with per-side truncation
        enc_q = self.tokenizer(
            queries,
            add_special_tokens=False,
            truncation=(q_max is not None),
            max_length=q_max,
        )
        enc_d = self.tokenizer(
            docs,
            add_special_tokens=False,
            truncation=(d_max is not None),
            max_length=d_max,
        )

        q_ids_list = enc_q["input_ids"]
        d_ids_list = enc_d["input_ids"]

        cls_id = self.tokenizer.cls_token_id or self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        pad_id = (
            self.tokenizer.pad_token_id
            if getattr(self.tokenizer, "pad_token_id", None) is not None
            else (self.tokenizer.eos_token_id if getattr(self.tokenizer, "eos_token_id", None) is not None else 0)
        )

        built_inputs: List[List[int]] = []
        lengths: List[int] = []
        for q_ids, d_ids in zip(q_ids_list, d_ids_list):
            # assemble with special tokens
            # desired: [CLS] q_ids [SEP] d_ids [SEP]
            total_len = 1 + len(q_ids) + 1 + len(d_ids) + 1
            if total_len > combined_limit:
                overflow = total_len - combined_limit
                # prefer truncating document side first
                if len(d_ids) > overflow:
                    d_ids = d_ids[: len(d_ids) - overflow]
                else:
                    # drop all doc tokens and remove remaining from query
                    needed = overflow - len(d_ids)
                    d_ids = []
                    if needed >= len(q_ids):
                        q_ids = []
                    else:
                        q_ids = q_ids[: len(q_ids) - needed]

            inp = [cls_id] + q_ids + [sep_id] + d_ids + [sep_id]
            built_inputs.append(inp)
            lengths.append(len(inp))

        # pad to batch max length (bounded by combined_limit)
        batch_max = min(combined_limit, max(lengths) if lengths else 0)
        padded_inputs: List[List[int]] = []
        attention_masks: List[List[int]] = []
        token_type_ids_batch: List[List[int]] = []
        for inp in built_inputs:
            if len(inp) > batch_max:
                seq = inp[:batch_max]
            else:
                seq = inp + [pad_id] * (batch_max - len(inp))
            padded_inputs.append(seq)
            attention_masks.append([1 if i < len(inp) else 0 for i in range(batch_max)])

            # token_type_ids: 0 for [CLS] + query + first [SEP], 1 for doc + final [SEP]
            # compute split point: index of first sep (after query)
            try:
                first_sep = inp.index(sep_id)
            except ValueError:
                first_sep = 1 + len(q_ids_list[0])  # fallback
            ttypes: List[int] = []
            for i in range(batch_max):
                if i <= first_sep:
                    ttypes.append(0)
                else:
                    ttypes.append(1)
            token_type_ids_batch.append(ttypes)

        ids_tensor = torch.tensor(padded_inputs, dtype=torch.long).to(self.device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        token_type_ids_tensor = torch.tensor(token_type_ids_batch, dtype=torch.long)

        return TokenizedTexts(
            None,
            ids_tensor,
            lengths_tensor,
            attention_mask_tensor if mask else None,
            token_type_ids_tensor,
        )

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):

        tokenized = self.batch_tokenize(inputs, maxlen=self.max_length, mask=True)
        # strange that some existing models on the huggingface don't use the token_type
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            result = self.model(
                tokenized.ids,
                token_type_ids= to_device(tokenized.token_type_ids, self.device),
                attention_mask= to_device(tokenized.mask, self.device),
            ).logits  # Tensor[float] of length records size
        return result

