"""Mixin for ModuleLoader subclasses that produce sentence-transformers
compatible checkpoints on HuggingFace Hub export.

Loaders that inherit from :class:`SpladeLoaderMixin` will
automatically write ST config files via
:meth:`~xpm_torch.module.ModuleLoader.write_hub_extras` and append ST
loading instructions via
:meth:`~xpm_torch.module.ModuleLoader.hub_readme_sections`.
"""

import json
from pathlib import Path


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
