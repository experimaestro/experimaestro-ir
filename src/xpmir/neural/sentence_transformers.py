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

from transformers import AutoConfig

from sentence_transformers import CrossEncoder

from experimaestro import Param, field, LightweightTask
from xpmir.text.huggingface.tokenizers import get_default_max_len
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

    pref_attn_implementation: Param[Optional[str]] = field(default=None)
    """Attention implementation to use (e.g. 'flash_attention_2', 'sdpa', or None)."""

    st_model: CrossEncoder

    def __post_init__(self):
        super().__post_init__()

        if self.max_length is None:
            # try to infer it from config
            self.max_length = get_default_max_len(self.model_id)
            logger.warning(
                f"No max_len provided for STCrossEncoder, using default hf: {self.max_length}"
            )

    def __initialize__(self):
        super().__initialize__()

        model_kwargs = {}
        if self.pref_attn_implementation:
            model_kwargs["attn_implementation"] = self.pref_attn_implementation
            logger.info(
                f"Using attention implementation: '{self.pref_attn_implementation}'"
            )
        else:
            logger.info("Using default attention implementation (None specified)")

        # Warn if no classification head is found
        try:
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            archs = getattr(config, "architectures", [])
            if not any("ForSequenceClassification" in arch for arch in archs):
                logger.warning(
                    f"No sequence classification head found in '{self.model_id}'. "
                    f"Are you sure you are loading a CrossEncoder? "
                    f"If you are fine-tuning from an encoder, use build_STCrossEncoder."
                )
        except Exception:
            pass

        self.st_model = CrossEncoder(
            self.model_id,
            max_length=self.max_length,
            model_kwargs=model_kwargs,
        )

        actual_attn = getattr(self.st_model.model.config, "_attn_implementation", None)
        if not (actual_attn and "flash" in actual_attn.lower()):
            logger.warning(
                f"FA2 may not be active (attn_impl={actual_attn!r}); training will be slower."
            )

        self._initialized = True

    def save_model(self, path: Path):
        """Save the model in native SentenceTransformers format."""
        if path.suffix == ".safetensors":
            # ModuleLoader's default tries to save a safetensors file directly.
            # We intercept it to save the whole directory instead.
            path = path.parent
        self.st_model.save(str(path))

    def load_model(self, path: Path):
        """Load the model from a native SentenceTransformers directory."""
        if path.is_file():
            path = path.parent
        self.model_id = str(path)
        self.st_model = CrossEncoder(self.model_id)
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

        # Route through the ST Transformer module's preprocess so any prompt /
        # chat template (e.g. Qwen3/mxbai generative rerankers) is applied
        # identically to predict().
        max_length = options.max_length if options else self.max_length
        processing_kwargs = (
            {"text": {"max_length": max_length, "truncation": True}}
            if max_length
            else None
        )
        r = self.st_model[0].preprocess(pairs, processing_kwargs=processing_kwargs)

        standard_keys = {"input_ids", "length", "attention_mask", "token_type_ids"}
        kwargs = {k: v for k, v in r.items() if k not in standard_keys}

        return TokenizedTexts(
            tokens=None,
            ids=r["input_ids"],
            lens=r.get("length", None),
            mask=r.get("attention_mask", None),
            token_type_ids=r.get("token_type_ids", None),
            kwargs=kwargs if kwargs else None,
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
        inputs: Optional[BaseItems] = None,
        tokenized: Optional[TokenizedTexts] = None,
    ):
        if not self._initialized:
            self.initialize()

        if tokenized is None:
            assert inputs is not None, "Either inputs or tokenized must be provided"
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

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            features = {
                "input_ids": to_device(tokenized.ids, self.device),
            }
            if tokenized.mask is not None:
                features["attention_mask"] = to_device(tokenized.mask, self.device)
            if tokenized.token_type_ids is not None:
                features["token_type_ids"] = to_device(
                    tokenized.token_type_ids, self.device
                )
            if getattr(tokenized, "kwargs", None) is not None:
                for k, v in tokenized.kwargs.items():
                    features[k] = (
                        to_device(v, self.device) if isinstance(v, torch.Tensor) else v
                    )

            # Run the ST module pipeline so that CausalLM-based generative
            # rerankers (5.4+) get their LogitScore reduction applied;
            # classification CEs keep the same head-logit semantics.
            output = self.st_model(features)
            result = output["scores"]

            # predict() returns scores without a trailing singleton dim
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
    pref_attn_implementation: Optional[str] = None,
) -> Tuple[STCrossEncoder, List[LightweightTask]]:
    """Creates an STCrossEncoder model.

    :param model_id: The HuggingFace model ID
    :param max_length: Maximum sequence length
    :returns: (STCrossEncoder, init_tasks)
    """
    default_max_len = get_default_max_len(model_id)

    if max_length and default_max_len > max_length:
        max_len = max_length
    else:
        logging.warning(
            f"No max_length provided or default max_length {default_max_len} is not greater than provided max_length {max_length}."
            f"Using default max_len {default_max_len} for CrossEncoder {model_id}"
        )
        max_len = None
    scorer = STCrossEncoder.C(
        model_id=model_id,
        max_length=max_len,
        pref_attn_implementation=pref_attn_implementation,
    )
    return scorer, [InitSTCrossEncoder.C(model=scorer)]
