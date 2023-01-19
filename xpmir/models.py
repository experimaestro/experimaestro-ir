from experimaestro.huggingface import ExperimaestroHFHub
from typing import Optional
from xpmir.neural.dual import DotDense
from xpmir.utils.utils import easylog
import importlib

logger = easylog()


def get_class(name: str):
    module_name, class_name = name.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class AutoModel:
    @staticmethod
    def load_from_hf_hub(hf_id_or_folder: str, variant: Optional[str] = None):
        """Loads from hugging face hub or from a folder"""
        return ExperimaestroHFHub.from_pretrained(hf_id_or_folder, variant=variant)

    @staticmethod
    def sentence_scorer(hf_id: str):
        """Loads from hugging face hub using a sentence transformer"""
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            logger.error(
                "Sentence transformer is not installed: pip install -U sentence_transformers"
            )
            raise

        encoder = SentenceTransformer(hf_id)
        return DotDense(encoder=encoder)
