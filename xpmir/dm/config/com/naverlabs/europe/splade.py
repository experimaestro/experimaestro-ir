from pathlib import Path
from experimaestro import Param
from datamaestro.definitions import datatags, dataset
from datamaestro.data import Base
from datamaestro.download.single import filedownloader

BASEURL = "https://raw.githubusercontent.com/naver/splade/main/weights"


class HuggingfaceConfiguration(Base):
    """Huggingface serialized model"""

    configdir: Param[Path]

    def auto(self, cls):
        """Proxy method to get the model by passing the class ``cls``

        Examples:
          >>> model = config.auto(AutoModel)
        """
        return cls.from_pretrained(self.configdir)


def splademodel(modelname):
    def annotate(method):
        for name in [
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        ]:
            method = filedownloader(name, f"{BASEURL}/{modelname}/{name}")(method)
        method = datatags("information retrieval")(method)

        method = filedownloader(
            "pytorch_model.bin",
            f"https://media.githubusercontent.com/media/naver/splade/main/weights/{modelname}/pytorch_model.bin",
        )(method)

        return method

    return annotate


@splademodel("distilsplade_max")
@dataset(HuggingfaceConfiguration, url="https://github.com/naver/splade")
def distilsplade_max(config, **kwargs):
    """Splade v2 model (Distilled with max aggregation)"""
    return {"configdir": config.parent}
