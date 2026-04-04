import io
import os
from pathlib import Path
from typing import Optional, Union, Dict
import shutil
from experimaestro import Config, Param, field
from xpmir.neural.dual import DotDense
from xpmir.neural.huggingface import HFCrossScorer
from xpm_torch import ModuleLoader
from xpm_torch.actions import ExportAction
from xpm_torch.huggingface import TorchHFHub
from xpm_torch.module import ReadmeSection

import logging

logger = logging.getLogger(__name__)


def get_class(name: str):
    module_name, class_name = name.split(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class XPMIRHFHub(TorchHFHub):
    """HF Hub integration for xpmir models.

    Extends :class:`~xpm_torch.huggingface.TorchHFHub` with xpmir-specific
    README sections (frontmatter, description, usage, results) and
    TensorBoard log copying.
    """

    def __init__(
        self,
        config: Config,
        *,
        doc: Optional[str] = None,
        bibtex: Optional[str] = None,
        model_id: Optional[str] = None,
        evaluations=None,
        model_key: Optional[str] = None,
        tb_logs: Optional[Dict[str, Path]] = None,
    ):
        super().__init__(config)
        self.doc = doc
        self.bibtex = bibtex
        self.model_id = model_id
        self.evaluations = evaluations
        self.model_key = model_key
        self.tb_logs = tb_logs

    def _xpmir_usage_section(self) -> str:
        return (
            "## Using the model\n\n"
            "The model can be loaded with [experimaestro "
            "IR](https://experimaestro-ir.readthedocs.io/en/latest/)\n\n"
            "To use in further experiments with XPMIR, load the model loader:\n"
            "```py\n"
            "from xpmir.models import AutoModel\n\n"
            f'loader = AutoModel.load_from_hf_hub("{self.model_id}")\n'
            "# loader.model is the model config\n"
            "# pass loader as an init task to load the weights\n"
            "```\n\n"
            "For direct inference:\n\n"
            "```py\n"
            "from xpmir.models import AutoModel\n\n"
            f'model = AutoModel.load_from_hf_hub("{self.model_id}", as_instance=True)\n'
            'model.rsv("walgreens store sales average", '
            '"The average Walgreens salary ranges...")\n'
            "```"
        )

    def _results_section(self) -> str:
        out = io.StringIO()
        out.write("## Results\n\n")
        self.evaluations.output_model_results(self.model_key, file=out)
        return out.getvalue()

    def _readme_base_sections(self):
        sections = [
            ReadmeSection("frontmatter", "---\nlibrary_name: xpmir\n---\n"),
        ]
        if self.doc:
            sections.append(ReadmeSection("description", f"{self.doc}\n"))
        if self.model_id:
            sections.append(ReadmeSection("usage", self._xpmir_usage_section()))
        if self.evaluations and self.model_key:
            sections.append(ReadmeSection("results", self._results_section()))
        if self.bibtex:
            sections.append(
                ReadmeSection(
                    "citation",
                    f"## Citation\n\n```bibtex\n{self.bibtex}\n```",
                )
            )
        return sections

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        super()._save_pretrained(save_directory)

        if self.tb_logs:
            runs_dir = save_directory / "runs"
            runs_dir.mkdir()
            for key, path in self.tb_logs.items():
                shutil.copytree(path, runs_dir / key)


class XPMIRExportAction(ExportAction):
    """Export action that uses XPMIRHFHub for xpmir-specific README sections."""

    doc: Param[str] = field(default="", ignore_default=True)
    """Paper description or title"""

    bibtex: Param[str] = field(default="", ignore_default=True)
    """BibTeX citation"""

    def get_hub(self):
        return XPMIRHFHub(self.loader, doc=self.doc or None, bibtex=self.bibtex or None)


class AutoModel:
    @staticmethod
    def load_from_hf_hub(
        hf_id_or_folder: str,
        as_instance: bool = False,
    ):
        """Loads a model from HuggingFace Hub or from a local folder.

        Returns a :class:`~xpm_torch.module.ModuleLoader`. Use
        ``loader.model`` to access the model config, and ``loader`` itself
        as an init task.

        If ``as_instance=True``, executes the loader and returns the
        ready-to-use model instance directly.
        """
        local_files_only = os.environ.get("HF_HUB_OFFLINE", False)
        loader = XPMIRHFHub.from_pretrained(
            hf_id_or_folder,
            local_files_only=local_files_only,
        )

        if not isinstance(loader, ModuleLoader):
            raise TypeError(f"Expected ModuleLoader, got {type(loader)}")

        if as_instance:
            loader.execute()
            return loader.model

        return loader

    @staticmethod
    def push_to_hf_hub(config: Config, *args, **kwargs):
        """Push to HuggingFace Hub

        See ModelHubMixin.push_to_hub for the other arguments
        """
        return XPMIRHFHub(config).push_to_hub(*args, **kwargs)

    @staticmethod
    def sentence_scorer(hf_id: str):
        """Loads from hugging face hub using a sentence transformer"""
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            logger.error(
                "Sentence transformer is not installed:"
                "pip install -U sentence_transformers"
            )
            raise

        encoder = SentenceTransformer(hf_id)
        return DotDense(encoder=encoder)

    @staticmethod
    def cross_encoder_model(hf_id: str, max_length: int = 512):
        """Loads from huggingface hub in to a form of a cross-scorer, it returns
        a sentence_transformer model for cross encoder"""

        scorer = HFCrossScorer(hf_id=hf_id, max_length=max_length)
        return scorer
