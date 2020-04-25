import importlib
from experimaestro_ir import NS

# Notes
# The different pipelines are implemented in onir.pipelines

openNIR = NS.openNIR


# --- Learning pipeline parameters


class Factory:
    """Generic factory method"""

    def create(self, **kwargs):
        module = importlib.import_module(f"{self.PACKAGE_NAME}.{self.CLASS_NAME}")
        factory = getattr(module, self.CLASS_NAME)

        config = factory.default_config()
        kwargs["config"] = config
        return factory(**kwargs)
