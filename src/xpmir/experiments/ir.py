from typing import Any, List, Optional, Dict
import logging
import pandas as pd
from pathlib import Path
import click
import io
from functools import cached_property

import docstring_parser
from experimaestro import RunMode, Config
from experimaestro.exceptions import HandledException
from xpmir.evaluation import EvaluationsCollection
from xpmir.models import XPMIRHFHub
from xpmir.papers.results import PaperResults
from xpmir.experiments.learning import LearningExperimentHelper
from xpmir.learning.optim import TensorboardService


class UploadToHub:
    def __init__(self, model_id: Optional[str], doc: docstring_parser.Docstring):
        self.model_id = model_id
        self.doc = doc

    def send_scorer(
        self,
        models: Dict[str, Config],
        *,
        evaluations: Optional[EvaluationsCollection] = None,
        tb_logs: Dict[str, Path],
        add_des: str = "",
    ):
        """Upload the scorer(s) to the HuggingFace Hub

        :param models: The models to upload, each with a key
        :param tb_logs: The tensorboard logs
        :param evaluations: Models evaluations, defaults to None
        :param add_des: Extra documentation, defaults to ""
        """
        if self.model_id is None:
            return

        out = io.StringIO()
        out.write(
            """---
library_name: xpmir
---
"""
        )

        out.write(f"{self.doc}\n\n")
        out.write(f"{add_des}\n\n")

        out.write("\n## Using the model")
        out.write(
            f"""
The model can be loaded with [experimaestro
IR](https://experimaestro-ir.readthedocs.io/en/latest/)

If you want to use the model in further experiments with XPMIR,
use this code:
```py
from xpmir.models import AutoModel
from xpmir.models import AutoModel

model, init_tasks = AutoModel.load_from_hf_hub("{self.model_id}")
```


Use this code if you want to use the model in inference only:

```py
from xpmir.models import AutoModel
from xpmir.models import AutoModel

model = AutoModel.load_from_hf_hub("{self.model_id}", as_instance=True)
model.rsv("walgreens store sales average", "The average Walgreens salary ranges...")
```
"""
        )

        assert len(models) == 1, "Cannot deal with more than one variant"
        ((key, model),) = list(models.items())

        if evaluations is not None:
            out.write("\n## Results\n\n")
            evaluations.output_model_results(key, file=out)

        readme_md = out.getvalue()

        logging.info("Uploading to HuggingFace Hub")
        XPMIRHFHub(model, readme=readme_md, tb_logs=tb_logs).push_to_hub(
            repo_id=self.model_id, config={}
        )


class IRExperimentHelper(LearningExperimentHelper):
    """Helper for IR experiments"""

    def run(self, extra_args: List[str], configuration: Any):
        @click.option("--upload-to-hub", type=str)
        @click.command()
        def cli(upload_to_hub: str):
            try:
                results = self.callable(self, configuration)
            except Exception as e:
                logging.exception("Error while running the experiment")
                raise HandledException(e)
            self.xp.wait()

            if isinstance(results, PaperResults) and self.xp.run_mode == RunMode.NORMAL:
                if upload_to_hub is not None:
                    if configuration.title == "" and configuration.description == "":
                        doc = docstring_parser.parse(self.callable.__doc__)
                    else:
                        doc = f"# {configuration.title}\n{configuration.description}"
                    upload = UploadToHub(upload_to_hub, doc)

                    upload.send_scorer(
                        results.models,
                        evaluations=results.evaluations,
                        tb_logs=results.tb_logs,
                    )

                # Print the results
                df = results.evaluations.to_dataframe()
                pd.set_option("display.max_columns", None)
                pd.set_option("display.max_rows", None)
                pd.set_option("display.width", 200)
                print(df)  # noqa: T201

                # And save them

                csv_path = self.xp.resultspath / "results.csv"
                if not self.xp.resultspath.exists():
                    self.xp.resultspath.mkdir(parents=True, exist_ok=True)
                logging.info(f"Saved results in {csv_path.absolute()}")
                with csv_path.open("wt") as fp:
                    df.to_csv(fp, index=False)

        return cli(extra_args, standalone_mode=False)

    @cached_property
    def tensorboard_service(self):
        return self.xp.add_service(
            TensorboardService(self.xp, self.xp.resultspath / "runs")
        )


ir_experiment = IRExperimentHelper.decorator
"""Uses an IR experiment helper that provides

1. Tensorboard service (from Learning)
1. Upload to HuggingFace
1. Printing the evaluation results
"""
