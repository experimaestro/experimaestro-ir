"""Mixin for ModuleLoader subclasses that produce sentence-transformers
compatible checkpoints on HuggingFace Hub export.

Loaders that inherit from :class:`SentenceTransformerLoaderMixin` will
automatically write ST config files via
:meth:`~xpm_torch.module.ModuleLoader.write_hub_extras` and append ST
loading instructions via
:meth:`~xpm_torch.module.ModuleLoader.hub_readme_extra`.
"""

import json
from pathlib import Path


class SentenceTransformerLoaderMixin:
    """Mixin for ModuleLoader subclasses that adds sentence-transformers
    compatibility on HF Hub export.

    Subclasses can set :attr:`st_model_type` (default: ``"SparseEncoder"``)
    and :attr:`st_similarity` (default: ``"dot"``).
    """

    st_model_type: str = "SparseEncoder"
    """The sentence-transformers model type."""

    st_similarity: str = "dot"
    """The similarity function name."""

    def write_hub_extras(self, save_directory: Path):
        """Write sentence-transformers config files for HF Hub export."""
        model_dir = save_directory / "model"
        encoder_dir = (
            model_dir / "encoder" if (model_dir / "encoder").exists() else model_dir
        )

        # Read vocab_size from the saved encoder config
        config_path = encoder_dir / "config.json"
        vocab_size = None
        if config_path.exists():
            with open(config_path) as f:
                vocab_size = json.load(f).get("vocab_size")

        has_query_encoder = (model_dir / "query_encoder").exists()

        # modules.json
        if has_query_encoder:
            modules = [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "encoder",
                    "type": f"sentence_transformers.models.{self.st_model_type}",
                },
            ]
        else:
            modules = [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": f"sentence_transformers.models.{self.st_model_type}",
                },
                {
                    "idx": 1,
                    "name": "1_SpladePooling",
                    "path": "1_SpladePooling",
                    "type": "sentence_transformers.models.SpladePooling",
                },
            ]

        (save_directory / "modules.json").write_text(json.dumps(modules, indent=2))

        # config_sentence_transformers.json
        st_config = {
            "prompts": {},
            "default_prompt_name": None,
            "similarity_fn_name": self.st_similarity,
            "model_type": self.st_model_type,
        }
        (save_directory / "config_sentence_transformers.json").write_text(
            json.dumps(st_config, indent=2)
        )

        # sentence_bert_config.json (for compat)
        sb_config = {
            "max_seq_length": 256,
            "do_lower_case": False,
        }
        (save_directory / "sentence_bert_config.json").write_text(
            json.dumps(sb_config, indent=2)
        )

        if not has_query_encoder:
            pooling_dir = save_directory / "1_SpladePooling"
            pooling_dir.mkdir(exist_ok=True)
            pooling_config = {
                "pooling_strategy": "max",
                "activation_function": "log1p_relu",
                "word_embedding_dimension": vocab_size,
            }
            (pooling_dir / "config.json").write_text(
                json.dumps(pooling_config, indent=2)
            )

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
