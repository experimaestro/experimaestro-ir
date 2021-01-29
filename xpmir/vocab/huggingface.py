import torch
from experimaestro import param, config, Choices, Param

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

from xpmir.neural.modules import CustomBertModelWrapper
import xpmir.vocab as vocab


@config()
class TransformerVocab(vocab.Vocab):
    """
    Args:

    model_id: Model ID from huggingface
    trainable: Whether BERT parameters should be trained
    layer: Layer to use (0 is the last, -1 to use them all)
    """

    model_id: Param[str] = "bert-base-uncased"
    trainable: Param[bool] = False
    layer: Param[int] = 0

    CLS: int
    SEP: int

    def initialize(self):
        super().initialize()
        self.model = AutoModel.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        layer = self.layer
        if layer == -1:
            layer = None
        self.CLS = self.tok2id("[CLS]")
        self.SEP = self.tok2id("[SEP]")
        if self.trainable:
            self.model.train()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def tok2id(self, tok):
        return self.tokenizer.vocab[tok]

    def id2tok(self, idx):
        if torch.is_tensor(idx):
            if len(idx.shape) == 0:
                return self.id2tok(idx.item())
            return [self.id2tok(x) for x in idx]
        # return self.tokenizer.ids_to_tokens[idx]
        return self.tokenizer.id_to_token(idx)

    def lexicon_size(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()

    def maxtokens(self) -> int:
        return 512


@config()
class IndependentTransformerVocab(TransformerVocab):
    """Encodes as [CLS] QUERY [SEP]"""

    def _forward(self, in_toks, lens=None, seg_id=0):
        if lens is None:
            # if no lens provided, assume all are full length, I guess... not great
            lens = torch.full_like(in_toks[:, 0], in_toks.shape[1])
        maxlen = self.bert.config.max_position_embeddings
        MAX_TOK_LEN = maxlen - 2  # -2 for [CLS] and [SEP]
        toks, _ = util.subbatch(in_toks, MAX_TOK_LEN)
        mask = util.lens2mask(lens, in_toks.shape[1])
        mask, _ = util.subbatch(mask, MAX_TOK_LEN)
        toks = torch.cat([torch.full_like(toks[:, :1], self.CLS), toks], dim=1)
        toks = torch.cat([toks, torch.full_like(toks[:, :1], self.SEP)], dim=1)
        ONES = torch.ones_like(mask[:, :1])
        mask = torch.cat([ONES, mask, ONES], dim=1)
        segment_ids = torch.full_like(toks, seg_id)
        # Change -1 padding to 0-padding (will be masked)
        toks = torch.where(toks == -1, torch.zeros_like(toks), toks)
        result = self.bert(toks, segment_ids, mask)
        if not self.vocab.last_layer:
            cls_result = [r[:, 0] for r in result]
            result = [r[:, 1:-1, :] for r in result]
            result = [util.un_subbatch(r, in_toks, MAX_TOK_LEN) for r in result]
        else:
            BATCH = in_toks.shape[0]
            result = result[-1]
            cls_output = result[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH : (i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            result = result[:, 1:-1, :]
            result = util.un_subbatch(result, in_toks, MAX_TOK_LEN)
        return result, cls_result

    def forward(self, in_toks, lens):
        results, _ = self._forward(in_toks, lens)
        return results

    def enc_query_doc(self, **inputs):
        result = {}
        if "query_tok" in inputs and "query_len" in inputs:
            query_results, query_cls = self._forward(
                inputs["query_tok"], inputs["query_len"], seg_id=0
            )
            result.update({"query": query_results, "query_cls": query_cls})
        if "doc_tok" in inputs and "doc_len" in inputs:
            doc_results, doc_cls = self._forward(
                inputs["doc_tok"], inputs["doc_len"], seg_id=1
            )
            result.update({"doc": doc_results, "doc_cls": doc_cls})
        return result


@config()
class JointTransformer(TransformerVocab):
    """Encodes as [CLS] QUERY [SEP] DOCUMENT"""

    def enc_query_doc(self, **inputs):
        query_tok, query_len = inputs["query_tok"], inputs["query_len"]
        doc_tok, doc_len = inputs["doc_tok"], inputs["doc_len"]
        BATCH, QLEN = query_tok.shape
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - 3  # -3 [CLS] and 2x[SEP]

        doc_toks, sbcount = util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask = util.lens2mask(doc_len, doc_tok.shape[1])
        doc_mask, _ = util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = util.lens2mask(query_len, query_toks.shape[1])
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.CLS)
        SEPS = torch.full_like(query_toks[:, :1], self.SEP)
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat(
            [NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1
        )

        # Change -1 padding to 0-padding (will be masked)
        toks = torch.where(toks == -1, torch.zeros_like(toks), toks)

        result = self.bert(toks, segment_ids, mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1 : QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2 : -1] for r in result]
        doc_results = [
            util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results
        ]

        cls_results = []
        for layer in range(len(result)):
            cls_output = result[layer][:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH : (i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        if self.vocab.last_layer:
            query_results = query_results[-1]
            doc_results = doc_results[-1]
            cls_results = cls_results[-1]

        return {"query": query_results, "doc": doc_results, "cls": cls_results}
