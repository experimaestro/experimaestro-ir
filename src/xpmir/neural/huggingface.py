from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from experimaestro import DataPath, Param, LightweightTask

from xpmir.text import TokenizedTexts
from xpmir.letor.records import BaseItems
from xpmir.rankers import AbstractModuleScorer
from xpm_torch.module import ModuleLoader, ReadmeSection
from xpm_torch.utils import to_device

from xpmir.text.huggingface.base import (
    HFConfigID,
    HFSequenceClassification,
    HFModelInitBase,
    _resolve_model_path,
    is_local_files_only,
)
from xpmir.text.huggingface.tokenizers import HFTokenizer
from xpmir.text.tokenizers import TokenizerOptions

import logging

logger = logging.getLogger(__name__)


class HFQueryDocTokenizer(HFTokenizer):
    """Specific tokenizer for Cross-Scorers that handles query and document truncation.

    This tokenizer allows for independent limits on query and document lengths,
    while ensuring the combined sequence ([CLS] query [SEP] document [SEP])
    never exceeds the model's maximum length.

    Truncation strategy:

    1. Initial encoding caps each side at its respective max length (or the
       total available content limit).
    2. If the combined length still exceeds the total limit, the document is
       truncated first to make room.
    3. The query is only truncated if the document is entirely consumed and
       the sequence still exceeds the limit.

    This ensures that if a query is short, the document can utilize the
    remaining space up to the total limit.
    """

    max_query_length: Param[Optional[int]]
    """maximum number of tokens for the query side (defaults to max_doc_length // 2)"""

    max_doc_length: Param[Optional[int]]
    """maximum number of tokens for the document side (defaults to max_length)"""

    def __post_init__(self):
        super().__post_init__()

        # Sanity Check - max len should be set in parent class
        # Default behavior is doc_max_len = max_len | max_query_len = max_len // 2

        assert isinstance(self.max_length, int)

        if self.max_doc_length is None:
            logger.warning(
                f"No max_docs_len provided, using default max Len: {self.max_length}"
            )
            self.max_doc_length = self.max_length

        if self.max_query_length is None:
            self.max_query_length = self.max_doc_length // 2
            logger.warning(
                f"No max_query_len provided, using half of max Len: {self.max_query_length}"
            )

        assert isinstance(self.max_doc_length, int)
        assert isinstance(self.max_query_length, int)

    def tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        """Tokenize (query, document) pairs with maxlen for each side."""
        # Combined sequence length cap
        combined_limit = self.max_length
        if options and options.max_length is not None:
            combined_limit = min(options.max_length, combined_limit)

        num_special = 3  # [CLS] + [SEP] + [SEP]
        content_limit = combined_limit - num_special  # tokens available for text

        # determine per-side token limits for initial encoding
        # We use the tighter of (per-side limit, content_limit)
        q_max = min(self.max_query_length, content_limit)
        d_max = min(self.max_doc_length, content_limit)

        # get indexes of query/document pairs from the records
        ix_qs, ix_ds = input_records.pairs()

        queries = [input_records.unique_topics[i]["text_item"].text for i in ix_qs]
        docs = [input_records.unique_documents[i]["text_item"].text for i in ix_ds]

        # Special token IDs (BERT-style= [CLS] q [SEP] d [SEP])
        cls_id = self.tokenizer.cls_token_id  # [CLS]
        sep_id = self.tokenizer.sep_token_id  # [SEP]
        pad_id = self.tokenizer.pad_token_id

        # Verify the tokenizer has the tokens we expect
        assert cls_id is not None and sep_id is not None, (
            "Tokenizer must define cls_token and sep_token for pair encoding."
        )

        def _encode(texts: List[str], max_tokens: int) -> List[torch.Tensor]:
            enc = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_tensors=None,  # keep as List[List[int]] — lengths differ
            )
            # Convert each sample to a 1-D tensor individually (lengths vary)
            return [torch.tensor(ids, dtype=torch.long) for ids in enc["input_ids"]]

        query_tensors = _encode(queries, q_max)
        doc_tensors = _encode(docs, d_max)

        # Assemble [CLS] q [SEP] d [SEP], respecting combined_limit
        cls = torch.tensor([cls_id], dtype=torch.long)
        sep = torch.tensor([sep_id], dtype=torch.long)

        sequences: List[torch.Tensor] = []
        lengths: List[int] = []
        final_q_lengths: List[int] = []

        for q_ids, d_ids in zip(query_tensors, doc_tensors):
            # Trim doc then query if combined still overflows
            # Since we capped initial encoding at content_limit, q_ids + d_ids
            # might still be > content_limit, but each is <= content_limit.
            overflow = (q_ids.size(0) + d_ids.size(0)) - content_limit
            if overflow > 0:
                # 1. Trim document first
                d_trim = min(d_ids.size(0), overflow)
                d_ids = d_ids[: d_ids.size(0) - d_trim]
                overflow -= d_trim

            if overflow > 0:
                # 2. Trim query if still overflowing
                q_trim = min(q_ids.size(0), overflow)
                q_ids = q_ids[: q_ids.size(0) - q_trim]

            seq = torch.cat([cls, q_ids, sep, d_ids, sep])
            sequences.append(seq)
            lengths.append(seq.size(0))
            final_q_lengths.append(q_ids.size(0) + 2)  # [CLS] + q + [SEP]

        # Pad to max length in batch using F.pad
        max_len = max(lengths)
        input_ids = torch.stack(
            [F.pad(seq, (0, max_len - seq.size(0)), value=pad_id) for seq in sequences]
        )  # (B, max_len)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        attention_mask = (
            (torch.arange(max_len).unsqueeze(0) < lengths_tensor.unsqueeze(1)).long()
            if (options is None or options.return_mask)
            else None
        )  # (B, max_len)

        # token_type_ids: 0 for [CLS]+q+[SEP], 1 for d+[SEP]
        q_lengths_tensor = torch.tensor(final_q_lengths)
        token_type_ids = (
            torch.arange(max_len).unsqueeze(0) >= q_lengths_tensor.unsqueeze(1)
        ).long()  # 0 for query side, 1 for doc side

        return TokenizedTexts(
            None,
            input_ids,
            lengths,  # pre-padding per-sample lengths
            attention_mask if attention_mask is not None else None,
            token_type_ids if token_type_ids is not None else None,
        )


class LLMRankerTokenizer(HFTokenizer):
    """Specific tokenizer for LLM Cross-Scorers that handles query and document truncation separately"""

    max_query_length: Param[Optional[int]]
    """maximum number of tokens for the query side"""

    max_doc_length: Param[Optional[int]]
    """maximum number of tokens for the document side"""

    prompt_template: Param[str] = "Query: {query} Document: {document} Relevant:"
    """Prompt template for the LLM"""

    def __post_init__(self):
        super().__post_init__()

        # Sanity Check - max len should be set in parent class
        assert isinstance(self.max_length, int)

        if self.max_doc_length is None:
            self.max_doc_length = self.max_length // 2

        if self.max_query_length is None:
            self.max_query_length = self.max_length // 2

    def __initialize__(self):
        super().__initialize__()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        # Determine per-side token limits
        q_max = self.max_query_length
        d_max = self.max_doc_length

        ix_qs, ix_ds = input_records.pairs()
        queries = [input_records.unique_topics[i]["text_item"].text for i in ix_qs]
        docs = [input_records.unique_documents[i]["text_item"].text for i in ix_ds]

        # Use the template to create the full text
        # We need to truncate queries and documents separately before putting them in the template
        # for maximum control.
        def _truncate(text, max_len):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            return self.tokenizer.decode(tokens)

        full_texts = [
            self.prompt_template.format(
                query=_truncate(q, q_max), document=_truncate(d, d_max)
            )
            for q, d in zip(queries, docs)
        ]

        r = self.tokenizer(
            full_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            return_length=True,
        )

        return TokenizedTexts(
            tokens=None,
            ids=r["input_ids"],
            lens=r["length"].tolist(),
            mask=r.get("attention_mask", None),
            token_type_ids=r.get("token_type_ids", None),
        )


class InitCEFromHFID(HFModelInitBase):
    """Load Cross-encoder weights from a HuggingFace Hub model ID.
    this is specific to this class: we need to ensure n_labels is 1.
    Uses ``model.config.hf_id`` to resolve the model.
    """

    def execute(self):
        hf_id = self.model.config.hf_id
        model_id_or_path = _resolve_model_path(hf_id, self.model.automodel)

        config = self.model.autoconfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            dtype=torch.float32,
            local_files_only=is_local_files_only(),
        )

        # ensure that num_labels is one for a Cross-encoder
        if hasattr(config, "num_labels"):
            config.num_labels = 1
        else:
            logger.warning(
                "no 'num_labels param found in config, check that classifier outputs one label"
            )
        self.model.hf_config = config

        logging.info(
            "Loading pretrained model from HF (%s) with %s.%s",
            hf_id,
            self.model.automodel.__module__,
            self.model.automodel.__name__,
        )
        with self._init_context(empty_init=True):
            self.model.model = self.model.automodel.from_pretrained(
                model_id_or_path,
                config=config,
                trust_remote_code=True,
                local_files_only=is_local_files_only(),
            )
        self.model._initialized = True

    def hub_readme_sections(self) -> list:
        return []


class HFCrossScorer(AbstractModuleScorer):
    """Load a cross scorer model from the huggingface"""

    encoder: Param[HFSequenceClassification]
    """The encoder from Hugging Face"""

    tokenizer: Param[HFTokenizer]
    """The tokenizer for the cross-scorer"""

    def __initialize__(self):
        super().__initialize__()
        self.encoder.initialize()
        self.tokenizer.initialize()

    def batch_tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        """Transform the text to tokens by using the tokenizer"""
        return self.tokenizer.tokenize(input_records, options=options)

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    def forward(
        self,
        inputs: BaseItems,
        tokenized: Optional[TokenizedTexts] = None,
    ):
        if tokenized is None:
            tokenized = self.batch_tokenize(inputs)

        # strange that some existing models on the huggingface don't use the token_type
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            # to_device are no op here as wrapped with fabric and already on the right device,
            # but ensures compatibility if the model is used outside of fabric
            result = self.encoder.model(
                to_device(tokenized.ids, self.device),
                attention_mask=to_device(tokenized.mask, self.device),
                token_type_ids=to_device(tokenized.token_type_ids, self.device),
            ).logits  # Tensor[float] of length records size
        return result

    def save_model(self, path: Path):
        """Save the HF model and tokenizer in standard pretrained format."""
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.model.save_pretrained(path)
        self.tokenizer.tokenizer.save_pretrained(path)

    def load_model(self, path: Path):
        """Load from HF pretrained format."""
        from transformers import AutoModelForSequenceClassification

        self.encoder.model = AutoModelForSequenceClassification.from_pretrained(path)

    def loader_config(self, path: Path, *, settings=None) -> "CrossEncoderModuleLoader":
        return CrossEncoderModuleLoader.C(
            value=self, encoder_path=path, settings=settings
        )

    def export_action(self, loader, **kwargs):
        from xpmir.models import XPMIRExportAction

        if self.doc:
            kwargs.setdefault("doc", self.doc)
        if self.bibtex:
            kwargs.setdefault("bibtex", self.bibtex)
        return XPMIRExportAction.C(loader=loader, **kwargs)


class CrossEncoderModuleLoader(ModuleLoader):
    """ModuleLoader for cross-encoder models.

    Saves the model in standard HuggingFace format (config.json +
    model.safetensors + tokenizer), which is directly loadable by
    sentence-transformers ``CrossEncoder``.
    """

    encoder_path: DataPath
    """Path to the encoder checkpoint directory"""

    def execute(self):
        self.value.initialize()
        self.value.load_model(Path(self.encoder_path))

    def hub_readme_sections(self) -> list:
        return [
            ReadmeSection(
                key="quick_loading",
                content=(
                    "## Loading with sentence-transformers\n\n"
                    "```python\n"
                    "from sentence_transformers import CrossEncoder\n\n"
                    'model = CrossEncoder("YOUR_ORG/YOUR_MODEL")\n'
                    "```"
                ),
                before="usage",
            ),
        ]


def hf_cross_scorer(
    hf_id: str,
    max_query_length: Optional[int] = None,
    max_doc_length: Optional[int] = None,
) -> Tuple[HFCrossScorer, List[LightweightTask]]:
    """Creates an HFCrossScorer model from a pre-trained HuggingFace checkpoint.
    if no max_query_length or max_doc_length is provided, will default to HF config max_length for qeur with no query truncation.
    :param hf_id: The HuggingFace model ID
    :param max_query_length: Maximum query length
    :param max_doc_length: Maximum document length
    :returns: (model, init_tasks) tuple
    """

    encoder = HFSequenceClassification.C(config=HFConfigID.C(hf_id=hf_id))
    init_tasks = [InitCEFromHFID.C(model=encoder)]
    tokenizer = HFQueryDocTokenizer.C(
        model_id=hf_id,
        max_query_length=max_query_length,
        max_doc_length=max_doc_length,
    )

    scorer = HFCrossScorer.C(encoder=encoder, tokenizer=tokenizer)
    return scorer, init_tasks
