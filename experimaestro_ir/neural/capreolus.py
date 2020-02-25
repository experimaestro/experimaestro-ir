import importlib
from datamaestro_text.data.ir import Adhoc
from datamaestro_text.data.trec import AdhocTopics, TrecAdhocResults
from experimaestro import config, task, argument
from experimaestro_ir import NS
from experimaestro_ir.models import Model
from experimaestro_ir.evaluation import TrecAdhocResults
from typing import List

import capreolus.collection
import capreolus.benchmark

CNS = NS.capreolus


@config(CNS.model)
class CapreolusModel:
    def create(self, collection, benchmark):
        module = importlib.import_module(f"capreolus.reranker.{self.CAPREOLUS_NAME}")
        factory = getattr(module, self.CAPREOLUS_NAME)
        config = {name: getattr(self, name) for name in factory.required_params()}

        # From pipeline.py
        # self.reranker = self.module2cls["reranker"](
        #   self.extractors[0].embeddings, self.benchmark.reranking_runs[cfg["fold"]], cfg
        # )

        self.extractors = []
        for cls in factory.EXTRACTORS:
            cfg = {}
            extractor = cls(
                None,  # self.cache_path,
                None,  # extractor_cache_dir,
                None,  # self.cfg,
                benchmark=benchmark,  # benchmark=self.benchmark,
                collection=collection,  # collection=self.collection,
                index=None,  # index=self.index,
            )
            extractor.build_from_benchmark(embeddings="glove6b", keepstops=False)
            self.extractors.append(extractor)

        embeddings = None
        return factory(self.extractors[0].embeddings, None, config)


# --- DRMM


@argument("nbins", default=29, help="number of bins in matching histogram")
@argument(
    "nodes", default=5, help="hidden layer dimension for feed forward matching network"
)
@argument("histType", default="LCH", help="histogram type: 'CH', 'NH' or 'LCH'")
@argument("gateType", default="IDF", help="term gate type: 'TV' or 'IDF'")
@config(CNS.model.drmm)
class DRMM(CapreolusModel):
    CAPREOLUS_NAME = "DRMM"


# --- Collection


class CapreolusCollection(capreolus.collection.Collection):
    def __init__(self, collections: List[Adhoc]):
        assert len(collections) == 1, "Can cope with one collection only at the moment"
        self.collections = collections


class CapreolusBenchmark(capreolus.benchmark.Benchmark):
    def __init__(self):
        pass


# --- Learning pipeline parameters


@argument(
    "maxdoclen",
    default=800,
    help="maximum document length (in number of terms after tokenization)",
)
@argument(
    "maxqlen",
    default=4,
    help="maximum query length (in number of terms after tokenization)",
)
@argument("batch", default=32, help="batch size")
@argument("niters", default=150, help="number of iterations to train for")
@argument(
    "itersize",
    default=4096,
    help="number of training instances in one iteration (epoch)",
)
@argument(
    "gradacc",
    default=1,
    help="number of batches to accumulate over before updating weights",
)
@argument("lr", default=0.001, help="learning rate")
@argument("seed", default=123_456, help="random seed to use")
@argument("sample", default="simple")
@argument(
    "softmaxloss",
    default=True,
    help="True to use softmax loss (over pairs) or False to use hinge loss",
)
@config()
class Pipeline:
    pass


@argument("device", default="cpu", help="Device")
@argument("dataparallel", default=False, help="Whether use data paralellism")
@argument("training", type=List[Adhoc], help="Collections to be used")
@argument("model", type=CapreolusModel, help="The model to be trained")
@task(CNS.learn)
class ModelLearn:
    def execute(self):
        # Adapted from train.py#train  in capreolus

        # Build a Capreolus collection and benchmark from training
        collection = CapreolusCollection(self.training)
        benchmark = CapreolusBenchmark()

        reranker = self.model.create(collection, benchmark)

        post_pipeline_init_time = time.time()
        run_path = os.path.join(pipeline.reranker_path, fold)
        logger.info("initialized pipeline with results path: %s", run_path)
        post_pipeline_init_time = time.time()
        info_path = os.path.join(run_path, "info")
        os.makedirs(info_path, exist_ok=True)
        weight_path = os.path.join(run_path, "weights")
        os.makedirs(weight_path, exist_ok=True)
        predict_path = os.path.join(run_path, "predict")

        reranker.to(device)
        if dataparallel:
            if pipeline.device == "gpu":
                if torch.cuda.device_count() > 1:
                    reranker.model = torch.nn.DataParallel(reranker.model)
                else:
                    logger.warning(
                        "ignoring dataparallel=gpu because only %s CUDA device(s) can be found",
                        torch.cuda.device_count(),
                    )
            else:
                logger.warning("Cannot setup data parallel for device %s", device)
        optimizer = reranker.get_optimizer()

        prepare_batch = functools.partial(
            _prepare_batch_with_strings, device=pipeline.device
        )
        datagen = benchmark.training_tuples(fold["train_qids"])

        # folds to predict on
        pred_folds = {}
        pred_fold_sizes = {}
        # prepare generators
        if pipeline.cfg["predontrain"]:
            pred_fold_sizes[pipeline.cfg["fold"]] = sum(
                1 for qid in fold["train_qids"] for docid in benchmark.pred_pairs[qid]
            )
            pred_folds[pipeline.cfg["fold"]] = (
                fold["train_qids"],
                predict_generator(pipeline.cfg, fold["train_qids"], benchmark),
            )

        for pred_fold, pred_qids in fold["predict"].items():
            pred_fold_sizes[pred_fold] = sum(
                1 for qid in pred_qids for docid in benchmark.pred_pairs[qid]
            )
            pred_folds[pred_fold] = (
                pred_qids,
                predict_generator(pipeline.cfg, pred_qids, benchmark),
            )

        metrics = {}
        initial_iter = 0
        history = []
        batches_since_update = 0
        dev_ndcg_max = -1
        batches_per_epoch = pipeline.cfg["itersize"] // pipeline.cfg["batch"]
        batches_per_step = pipeline.cfg.get("gradacc", 1)
        pbar_loop = tqdm(
            desc="loop",
            total=pipeline.cfg["niters"],
            initial=initial_iter,
            position=0,
            leave=True,
            smoothing=0.0,
        )
        pbar_train = tqdm(
            desc="training",
            total=pipeline.cfg["niters"] * pipeline.cfg["itersize"],
            initial=initial_iter * pipeline.cfg["itersize"],
            unit_scale=True,
            position=1,
            leave=True,
        )
        dev_best_info = ""
        logger.info(
            "It took {0} seconds to reach training loop after pipeline init".format(
                post_pipeline_init_time - time.time()
            )
        )
        pbar_info = tqdm(position=2, leave=True, bar_format="{desc}")
        for niter in range(initial_iter, pipeline.cfg["niters"]):
            reranker.model.train()
            reranker.next_iteration()
            iter_loss = []

            for bi, data in enumerate(datagen):
                data = prepare_batch(data)
                pbar_train.update(pipeline.cfg["batch"])

                tag_scores = reranker.score(data)
                loss = pipeline.lossf(
                    tag_scores[0], tag_scores[1], pipeline.cfg["batch"]
                )
                # loss /= batches_per_step
                iter_loss.append(loss.item())
                loss.backward()
                batches_since_update += 1
                if batches_since_update >= batches_per_step:
                    batches_since_update = 0
                    optimizer.step()
                    optimizer.zero_grad()

                if (bi + 1) % batches_per_epoch == 0:
                    break

            avg_loss = np.mean(iter_loss)
            pbar_info.set_description_str(
                f"loss: {avg_loss:0.5f}\t{dev_best_info}{'':40s}"
            )
            # logger.info("epoch = %d loss = %f", niter, avg_loss)

            # make predictions
            reranker.model.eval()
            for pred_fold, (pred_qids, pred_gen) in pred_folds.items():
                pbar_info.set_description_str(
                    f"loss: {avg_loss:0.5f}\t{dev_best_info}\t[predicting {pred_fold_sizes[pred_fold]} pairs]"
                )
                pred_gen = iter(pred_gen)
                predfn = os.path.join(predict_path, pred_fold, str(niter))
                os.makedirs(os.path.dirname(predfn), exist_ok=True)

                preds = predict_and_save_to_file(
                    pred_gen, reranker, predfn, prepare_batch
                )
                missing_qids = set(
                    qid for qid in pred_qids if qid not in preds or len(preds[qid]) == 0
                )
                if len(missing_qids) > 0:
                    raise RuntimeError(
                        "predictions for some qids are missing, which may cause trec_eval's output to be incorrect\nmissing qids: %s"
                        % missing_qids
                    )

                test_qrels = {
                    qid: labels
                    for qid, labels in pipeline.collection.qrels.items()
                    if qid in pred_qids
                }
                fold_metrics = eval_preds_niter(test_qrels, preds, niter)

                if (
                    pred_fold == "dev"
                    and fold_metrics["ndcg_cut_20"][1] >= dev_ndcg_max
                ):
                    dev_ndcg_max = fold_metrics["ndcg_cut_20"][1]
                    # logger.info("saving best dev model with dev ndcg@20: %0.3f", dev_ndcg_max)
                    dev_best_info = "dev best: %0.3f on iter %s" % (dev_ndcg_max, niter)
                    reranker.save(os.path.join(weight_path, "dev"))

                for metric, (x, y) in fold_metrics.items():
                    metrics.setdefault(pred_fold, {}).setdefault(metric, []).append(
                        (x, y)
                    )

            pbar_loop.update(1)
            pbar_info.set_description_str(
                f"loss: {avg_loss:0.5f}\t{dev_best_info}{'':40s}"
            )
            history.append((niter, avg_loss))

        pbar_train.close()
        pbar_loop.close()
        pbar_info.close()

        logger.info(dev_best_info)
        with open(os.path.join(info_path, "loss.txt"), "wt") as outf:
            for niter, loss in history:
                print("%s\t%s" % (niter, loss), file=outf)
                # _run.log_scalar("train.loss", loss, niter)

        plot_loss_curve(history, os.path.join(info_path, "loss.pdf"))
        plot_metrics(metrics, predict_path)

        with open("{0}/config.json".format(run_path), "w") as fp:
            json.dump(_config, fp)


@argument("base", type=Model)
@argument("model", type=ModelLearn)
@argument("topics", type=AdhocTopics)
@task(CNS.rerank)
class ModelRerank(TrecAdhocResults):
    def execute(self):
        pass
