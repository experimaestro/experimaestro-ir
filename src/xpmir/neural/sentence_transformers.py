"""Mixin for ModuleLoader subclasses that produce sentence-transformers
compatible checkpoints on HuggingFace Hub export.

Loaders that inherit from :class:`SpladeLoaderMixin` will
automatically write ST config files via
:meth:`~xpm_torch.module.ModuleLoader.write_hub_extras` and append ST
loading instructions via
:meth:`~xpm_torch.module.ModuleLoader.hub_readme_sections`.
"""

import json
import torch
from pathlib import Path
from typing import Optional, List, Tuple

from sentence_transformers import CrossEncoder
from sentence_transformers.base.modules import Transformer

from experimaestro import Param, field, LightweightTask
from xpmir.text import TokenizedTexts
from xpmir.letor.records import BaseItems
from xpmir.rankers import AbstractModuleScorer
from xpm_torch.utils import to_device
from xpmir.text.tokenizers import TokenizerOptions, TokenizerBase


import logging

logger = logging.getLogger(__name__)


MLM_TYPE = "sentence_transformers.sparse_encoder.models.MLMTransformer.MLMTransformer"
POOLING_TYPE = "sentence_transformers.sparse_encoder.models.SpladePooling.SpladePooling"
ROUTER_TYPE = "sentence_transformers.models.Router"


class SpladeLoaderMixin:
    """Mixin for ModuleLoader subclasses that adds sentence-transformers
    compatibility on HF Hub export.

    Expects the concrete loader to define ``encoder_path`` and optionally
    ``query_encoder_path`` DataPath fields. After experimaestro serialization,
    encoder files are at ``save_directory/encoder_path/`` and
    ``save_directory/query_encoder_path/``.

    Supports both symmetric (single encoder + SpladePooling) and asymmetric
    (Router with separate query/document pipelines) layouts.

    Subclasses can set :attr:`st_similarity` (default: ``"dot"``).
    """

    st_similarity: str = "dot"
    """The similarity function name."""

    def write_hub_extras(self, save_directory: Path):
        """Write sentence-transformers config files for HF Hub export."""
        # Directory names match __xpm_serialize__ mapping in SpladeModuleLoader
        DOC_DIR = "document_0_MLMTransformer"
        QUERY_DIR = "query_0_MLMTransformer"

        encoder_dir = save_directory / DOC_DIR
        has_query_encoder = (save_directory / QUERY_DIR).exists()

        # Read vocab_size from the saved encoder config
        config_path = encoder_dir / "config.json"
        vocab_size = None
        if config_path.exists():
            with open(config_path) as f:
                vocab_size = json.load(f).get("vocab_size")

        if has_query_encoder:
            self._write_asymmetric(save_directory, vocab_size)
        else:
            self._write_symmetric(save_directory, vocab_size)

        # config_sentence_transformers.json
        st_config = {
            "model_type": "SparseEncoder",
            "prompts": {},
            "default_prompt_name": None,
            "similarity_fn_name": self.st_similarity,
        }
        (save_directory / "config_sentence_transformers.json").write_text(
            json.dumps(st_config, indent=2)
        )

    def _write_symmetric(self, save_directory: Path, vocab_size):
        """Symmetric model: MLMTransformer + SpladePooling."""
        DOC_DIR = "document_0_MLMTransformer"
        modules = [
            {"idx": 0, "name": "0", "path": DOC_DIR, "type": MLM_TYPE},
            {
                "idx": 1,
                "name": "1",
                "path": "1_SpladePooling",
                "type": POOLING_TYPE,
            },
        ]
        (save_directory / "modules.json").write_text(json.dumps(modules, indent=2))

        # SpladePooling config
        pooling_dir = save_directory / "1_SpladePooling"
        pooling_dir.mkdir(exist_ok=True)
        (pooling_dir / "config.json").write_text(
            json.dumps(
                {
                    "pooling_strategy": "max",
                    "activation_function": "relu",
                    "word_embedding_dimension": vocab_size,
                },
                indent=2,
            )
        )

        # sentence_bert_config.json
        (save_directory / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": 256, "do_lower_case": False}, indent=2)
        )

    def _write_asymmetric(self, save_directory: Path, vocab_size):
        """Asymmetric model: Router with query/document pipelines."""
        DOC_DIR = "document_0_MLMTransformer"
        QUERY_DIR = "query_0_MLMTransformer"

        # modules.json: single Router entry
        modules = [
            {"idx": 0, "name": "0", "path": "", "type": ROUTER_TYPE},
        ]
        (save_directory / "modules.json").write_text(json.dumps(modules, indent=2))

        # router_config.json
        router_config = {
            "types": {
                QUERY_DIR: MLM_TYPE,
                "query_1_SpladePooling": POOLING_TYPE,
                DOC_DIR: MLM_TYPE,
                "document_1_SpladePooling": POOLING_TYPE,
            },
            "structure": {
                "query": [QUERY_DIR, "query_1_SpladePooling"],
                "document": [DOC_DIR, "document_1_SpladePooling"],
            },
            "parameters": {
                "default_route": "document",
                "allow_empty_key": True,
            },
        }
        (save_directory / "router_config.json").write_text(
            json.dumps(router_config, indent=2)
        )

        # SpladePooling configs for both routes
        pooling_config = {
            "pooling_strategy": "max",
            "activation_function": "relu",
            "word_embedding_dimension": vocab_size,
        }
        for pooling_name in ["query_1_SpladePooling", "document_1_SpladePooling"]:
            pooling_dir = save_directory / pooling_name
            pooling_dir.mkdir(exist_ok=True)
            (pooling_dir / "config.json").write_text(
                json.dumps(pooling_config, indent=2)
            )

        # sentence_bert_config.json in each encoder dir
        sb_config = {"max_seq_length": 256, "do_lower_case": False}
        for enc_dir in [DOC_DIR, QUERY_DIR]:
            p = save_directory / enc_dir / "sentence_bert_config.json"
            p.write_text(json.dumps(sb_config, indent=2))

    def hub_readme_sections(self):
        """Return ST loading section, positioned before XPMIR usage."""
        from xpm_torch.module import ReadmeSection

        return [
            ReadmeSection(
                key="quick_loading",
                content=(
                    "## Loading with sentence-transformers\n\n"
                    "This model is also compatible with the "
                    "[sentence-transformers](https://www.sbert.net/) library:\n\n"
                    "```python\n"
                    "from sentence_transformers import SparseEncoder\n\n"
                    'model = SparseEncoder("YOUR_ORG/YOUR_MODEL")\n'
                    "```"
                ),
                before="usage",
            ),
        ]


class STCrossEncoder(AbstractModuleScorer):
    """A cross-encoder model leveraging the sentence-transformers library.

    It supports both direct raw text input and pre-tokenized input, utilizing the
    ST model's native tokenization and forward pass for consistency.

    Example:
        >>> from xpmir.neural.sentence_transformers import st_cross_scorer
        >>> model, init_tasks = st_cross_scorer(model_id="mixedbread-ai/mxbai-rerank-base-v1")
    """

    model_id: Param[str]
    """The HuggingFace model ID or path."""

    max_length: Param[Optional[int]] = field(default=None)
    """Maximum sequence length for tokenization."""

    st_model: CrossEncoder

    def __initialize__(self):
        super().__initialize__()

        self.st_model = CrossEncoder(
            self.model_id,
            max_length=self.max_length,
        )
        self._initialized = True

    @property
    def tokenizer(self) -> TokenizerBase:
        """Returns a tokenizer that uses the ST model's native tokenization."""
        return self.st_model.tokenizer

    def tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        # Prepare raw texts
        queries = [record["text_item"].text for record in input_records.unique_topics]
        documents = [
            record["text_item"].text for record in input_records.unique_documents
        ]

        # Reconstruct pairs
        pairs = []
        q_ix, d_ix = input_records.pairs()
        for qi, di in zip(q_ix, d_ix):
            pairs.append([queries[qi], documents[di]])

        # Use ST model's preprocess to handle prompts and chat templates correctly
        max_length = options.max_length if options else self.max_length
        r = self.st_model.preprocess(pairs, max_length=max_length)

        return TokenizedTexts(
            tokens=None,
            ids=r["input_ids"],
            lens=r.get("length", None),
            mask=r.get("attention_mask", None),
            token_type_ids=r.get("token_type_ids", None),
        )

    def vocabulary_size(self) -> int:
        return self.st_model.tokenizer.vocab_size

    def tok2id(self, tok: str) -> int:
        return self.st_model.tokenizer.convert_tokens_to_ids(tok)

    def id2tok(self, idx: int) -> str:
        return self.st_model.tokenizer.convert_ids_to_tokens(idx)

    def batch_tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        """Transform the text to tokens by using the tokenizer."""
        return self.tokenize(input_records, options=options)

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    def forward(
        self,
        inputs: BaseItems,
        tokenized: Optional[TokenizedTexts] = None,
    ):
        if not self._initialized:
            self.initialize()

        if tokenized is None:
            # Extract raw pairs from inputs
            queries = [record["text_item"].text for record in inputs.unique_topics]
            documents = [record["text_item"].text for record in inputs.unique_documents]
            pairs = []
            q_ix, d_ix = inputs.pairs()
            for qi, di in zip(q_ix, d_ix):
                pairs.append([queries[qi], documents[di]])

            # Use raw ST predict function
            scores = self.st_model.predict(
                pairs,
                batch_size=len(pairs),
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            return to_device(scores, self.device)

        # If tokenized is given, use the model modules sequentially
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            features = {
                "input_ids": to_device(tokenized.ids, self.device),
                "attention_mask": to_device(tokenized.mask, self.device),
            }
            if tokenized.token_type_ids is not None:
                features["token_type_ids"] = to_device(
                    tokenized.token_type_ids, self.device
                )

            # Causal models need logits_to_keep=1 for LogitScore module
            if isinstance(self.st_model[0], Transformer) and self.st_model[
                0
            ].transformer_task in ("text-generation", "any-to-any"):
                features["logits_to_keep"] = 1

            # Run through all modules (Transformer, LogitScore, etc.)
            # using the model's native forward pass (BaseModel inherits nn.Sequential)
            features = self.st_model(features)
            result = features["scores"]

            # Squeeze [batch_size, 1] -> [batch_size] for single-label models
            # to match predict() behavior
            if result.ndim > 1 and result.shape[-1] == 1:
                result = result.squeeze(-1)

        return result


class InitSTCrossEncoder(LightweightTask):
    """Initializes the STCrossEncoder by loading the model."""

    model: Param[STCrossEncoder]

    def execute(self):
        self.model.initialize()


def st_cross_scorer(
    model_id: str,
    max_length: Optional[int] = None,
) -> Tuple[STCrossEncoder, List[LightweightTask]]:
    """Creates an STCrossEncoder model.

    :param model_id: The HuggingFace model ID
    :param max_length: Maximum sequence length
    :returns: (STCrossEncoder, init_tasks)
    """
    scorer = STCrossEncoder.C(
        model_id=model_id,
        max_length=max_length,
    )
    return scorer, [InitSTCrossEncoder.C(model=scorer)]
