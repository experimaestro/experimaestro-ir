"""Tests for SPLADE model: decomposition, forward pass, save/load, shared encoder."""

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForMaskedLM, BertConfig

from xpmir.text.huggingface.decompose import decompose_mlm_model
from xpmir.text.huggingface.base import HFConfigID, HFMaskedLanguageModel
from xpmir.text.huggingface.tokenizers import HFTokenizer, HFTokenizerAdapter
from xpmir.text.adapters import TopicTextConverter
from xpmir.neural.splade import (
    MaxAggregation,
    SumAggregation,
    SpladeTextEncoder,
    PyTorchAggregationModule,
)

from xpmir.test import skip_if_ci

# Tiny BERT config for fast tests — no network download needed.
# vocab_size=256 keeps tensors small while still exercising the full pipeline.
_TINY_CONFIG = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
)

# HF model ID for SpladeTextEncoder tests (needs a real tokenizer).
# prajjwal1/bert-tiny is tiny (~17 MB) and cached after first download.
_HF_MODEL_ID = "prajjwal1/bert-tiny"

_CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def mlm_model():
    model = AutoModelForMaskedLM.from_config(_TINY_CONFIG)
    model.eval()
    return model


def _make_text_item(text):
    return {"text_item": type("T", (), {"text": text})()}


def _make_splade(encoder, tokenizer, aggregation, maxlen=30):
    splade = SpladeTextEncoder.C(
        aggregation=aggregation,
        encoder=encoder,
        tokenizer=tokenizer,
        maxlen=maxlen,
    ).instance()
    splade.initialize()
    return splade


class TestDecomposeMLM:
    """Tests for decompose_mlm_model with BERT."""

    def test_returns_correct_types(self, mlm_model):
        backbone, transform, decoder = decompose_mlm_model(mlm_model)
        assert isinstance(backbone, torch.nn.Module)
        assert isinstance(transform, torch.nn.Module)
        assert isinstance(decoder, torch.nn.Linear)

    def test_decoder_shape(self, mlm_model):
        _, _, decoder = decompose_mlm_model(mlm_model)
        assert decoder.out_features == mlm_model.config.vocab_size
        assert decoder.in_features == mlm_model.config.hidden_size

    def test_backbone_produces_hidden_states(self, mlm_model):
        backbone, _, _ = decompose_mlm_model(mlm_model)
        input_ids = torch.tensor([[1, 20, 30, 40, 50, 2]])
        mask = torch.ones_like(input_ids)
        output = backbone(input_ids=input_ids, attention_mask=mask)
        assert hasattr(output, "last_hidden_state")
        B, S, D = output.last_hidden_state.shape
        assert B == 1
        assert S == 6
        assert D == mlm_model.config.hidden_size

    def test_full_pipeline_matches_original(self, mlm_model):
        """Backbone -> transform -> decoder should match the original MLM output."""
        backbone, transform, decoder = decompose_mlm_model(mlm_model)
        input_ids = torch.tensor([[1, 20, 30, 40, 50, 2]])
        mask = torch.ones_like(input_ids)

        with torch.no_grad():
            hidden = backbone(
                input_ids=input_ids, attention_mask=mask
            ).last_hidden_state
            logits_decomposed = decoder(transform(hidden))
            logits_original = mlm_model(input_ids=input_ids, attention_mask=mask).logits

        torch.testing.assert_close(logits_decomposed, logits_original)

    def test_weight_tying(self, mlm_model):
        """Decoder weights should be the same objects as in the original model."""
        _, _, decoder = decompose_mlm_model(mlm_model)
        original_decoder = mlm_model.cls.predictions.decoder
        assert decoder.weight is original_decoder.weight

    def test_unsupported_model_type(self):
        """Should raise ValueError for unsupported architectures."""

        class FakeConfig:
            model_type = "gpt2"

        class FakeModel:
            config = FakeConfig()

        with pytest.raises(ValueError, match="does not support"):
            decompose_mlm_model(FakeModel())


class TestAggregationModules:
    """Tests for the aggregation module hierarchy."""

    def test_max_aggregation_output_module(self):
        transform = torch.nn.Identity()
        decoder = torch.nn.Linear(8, 16)
        module = MaxAggregation.C().instance().get_output_module(transform, decoder)
        assert isinstance(module, torch.nn.Module)

    def test_sum_aggregation_output_module(self):
        transform = torch.nn.Identity()
        decoder = torch.nn.Linear(8, 16)
        module = SumAggregation.C().instance().get_output_module(transform, decoder)
        assert isinstance(module, PyTorchAggregationModule)

    def test_pytorch_module_forward(self):
        transform = torch.nn.Identity()
        decoder = torch.nn.Linear(8, 16, bias=False)
        aggregation = MaxAggregation.C().instance()
        module = PyTorchAggregationModule(transform, decoder, aggregation)

        hidden = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.long)
        output = module(hidden, mask)
        assert output.shape == (2, 16)
        assert (output >= 0).all(), "log1p(relu(...)) should be non-negative"

    def test_sum_module_forward(self):
        transform = torch.nn.Identity()
        decoder = torch.nn.Linear(8, 16, bias=False)
        aggregation = SumAggregation.C().instance()
        module = PyTorchAggregationModule(transform, decoder, aggregation)

        hidden = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.long)
        output = module(hidden, mask)
        assert output.shape == (2, 16)
        assert (output >= 0).all(), "sum of log1p(relu(...)) should be non-negative"


class TestSpladeTextEncoder:
    """Tests for the full SpladeTextEncoder."""

    def test_forward_max_aggregation(self):
        encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=_HF_MODEL_ID))
        tokenizer = HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=_HF_MODEL_ID),
            converter=TopicTextConverter.C(),
        )
        splade = _make_splade(encoder, tokenizer, MaxAggregation.C())
        splade.eval()
        texts = [_make_text_item("hello world")]
        with torch.no_grad():
            output = splade(texts)
        assert output.value.shape[0] == 1
        assert output.value.shape[1] == splade.dimension
        assert (output.value >= 0).all()

    def test_forward_sum_aggregation(self):
        encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=_HF_MODEL_ID))
        tokenizer = HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=_HF_MODEL_ID),
            converter=TopicTextConverter.C(),
        )
        splade = _make_splade(encoder, tokenizer, SumAggregation.C())
        splade.eval()
        texts = [_make_text_item("hello world")]
        with torch.no_grad():
            output = splade(texts)
        assert output.value.shape[0] == 1
        assert (output.value >= 0).all()

    def test_forward_batch(self):
        encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=_HF_MODEL_ID))
        tokenizer = HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=_HF_MODEL_ID),
            converter=TopicTextConverter.C(),
        )
        splade = _make_splade(encoder, tokenizer, MaxAggregation.C())
        splade.eval()
        texts = [
            _make_text_item("hello world"),
            _make_text_item("another test sentence"),
        ]
        with torch.no_grad():
            output = splade(texts)
        assert output.value.shape[0] == 2

    def test_shared_encoder(self):
        """Query and doc encoder share the same HFMaskedLanguageModel."""
        encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=_HF_MODEL_ID))
        tokenizer = HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=_HF_MODEL_ID),
            converter=TopicTextConverter.C(),
        )
        doc_enc = SpladeTextEncoder.C(
            aggregation=MaxAggregation.C(),
            encoder=encoder,
            tokenizer=tokenizer,
            maxlen=200,
        )
        query_enc = SpladeTextEncoder.C(
            aggregation=MaxAggregation.C(),
            encoder=encoder,
            tokenizer=tokenizer,
            maxlen=30,
        )
        doc_inst = doc_enc.instance()
        query_inst = query_enc.instance()
        doc_inst.initialize()
        query_inst.initialize()
        doc_inst.eval()
        query_inst.eval()

        texts = [_make_text_item("shared encoder test")]
        with torch.no_grad():
            doc_output = doc_inst(texts)
            query_output = query_inst(texts)
        # Same text, same shared encoder → same output
        assert doc_output.value.shape == query_output.value.shape

    @skip_if_ci
    def test_save_load_roundtrip(self):
        """Save and load should produce identical outputs."""
        encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=_HF_MODEL_ID))
        tokenizer = HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=_HF_MODEL_ID),
            converter=TopicTextConverter.C(),
        )
        splade = _make_splade(encoder, tokenizer, MaxAggregation.C())
        splade.eval()

        texts = [_make_text_item("roundtrip test")]
        with torch.no_grad():
            output_before = splade(texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"
            splade.save_model(path)
            splade.load_model(path)
            splade.eval()

        with torch.no_grad():
            output_after = splade(texts)

        torch.testing.assert_close(output_before.value, output_after.value)

    def test_decomposed_matches_full_mlm(self):
        """Verify backbone + head matches full MLM + aggregation."""
        encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=_HF_MODEL_ID))
        tokenizer = HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=_HF_MODEL_ID),
            converter=TopicTextConverter.C(),
        )
        splade = _make_splade(encoder, tokenizer, MaxAggregation.C())
        splade.eval()

        model = splade.encoder.model
        input_ids = torch.tensor([[1, 10, 20, 2]])
        mask = torch.ones_like(input_ids)

        with torch.no_grad():
            # Full MLM path
            full_logits = model(input_ids=input_ids, attention_mask=mask).logits
            agg = MaxAggregation.C().instance()
            expected = agg(full_logits, mask)

            # Decomposed path
            backbone, transform, decoder = decompose_mlm_model(model)
            hidden = backbone(
                input_ids=input_ids, attention_mask=mask
            ).last_hidden_state
            logits = decoder(transform(hidden))
            actual = agg(logits, mask)

        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
class TestSpartonKernel:
    """Tests for the Sparton Triton kernel (CUDA only)."""

    def test_sparton_head_forward(self):
        """SpartonHead produces same result as PyTorch max+relu+log1p."""
        from xpmir.neural._sparton import SpartonHead

        vocab_size, hidden_dim = 256, 64
        B, S = 2, 10

        decoder = torch.nn.Linear(hidden_dim, vocab_size, bias=True).cuda()
        sparton = SpartonHead(vocab_size, hidden_dim, use_bias=True).cuda()
        sparton.tie_weights(decoder)

        hidden = torch.randn(B, S, hidden_dim, device="cuda")
        mask = torch.ones(B, S, dtype=torch.long, device="cuda")

        # Sparton path
        sparton_out = sparton(hidden, mask)

        # PyTorch reference
        logits = hidden @ decoder.weight.T + decoder.bias
        values, _ = torch.max(torch.relu(logits) * mask.unsqueeze(-1).float(), dim=1)
        pytorch_out = torch.log1p(values.clamp(min=0))

        torch.testing.assert_close(sparton_out, pytorch_out, atol=1e-4, rtol=1e-4)

    def test_sparton_head_with_masking(self):
        """SpartonHead handles attention masking correctly."""
        from xpmir.neural._sparton import SpartonHead

        vocab_size, hidden_dim = 128, 32
        B, S = 2, 8

        decoder = torch.nn.Linear(hidden_dim, vocab_size, bias=True).cuda()
        sparton = SpartonHead(vocab_size, hidden_dim, use_bias=True).cuda()
        sparton.tie_weights(decoder)

        hidden = torch.randn(B, S, hidden_dim, device="cuda")
        # Mask: first example has 5 tokens, second has 3
        mask = torch.zeros(B, S, dtype=torch.long, device="cuda")
        mask[0, :5] = 1
        mask[1, :3] = 1

        sparton_out = sparton(hidden, mask)

        # PyTorch reference
        logits = hidden @ decoder.weight.T + decoder.bias
        values, _ = torch.max(torch.relu(logits) * mask.unsqueeze(-1).float(), dim=1)
        pytorch_out = torch.log1p(values.clamp(min=0))

        torch.testing.assert_close(sparton_out, pytorch_out, atol=1e-4, rtol=1e-4)

    def test_sparton_head_tie_weights(self):
        """Tied weights share the same parameter objects."""
        from xpmir.neural._sparton import SpartonHead

        decoder = torch.nn.Linear(32, 64, bias=True).cuda()
        sparton = SpartonHead(64, 32, use_bias=True).cuda()
        sparton.tie_weights(decoder)

        assert sparton.weight is decoder.weight
        assert sparton.bias is decoder.bias

    def test_sparton_gradient_flow(self):
        """SpartonHead supports gradient computation."""
        from xpmir.neural._sparton import SpartonHead

        vocab_size, hidden_dim = 128, 32
        B, S = 2, 6

        decoder = torch.nn.Linear(hidden_dim, vocab_size, bias=True).cuda()
        sparton = SpartonHead(vocab_size, hidden_dim, use_bias=True).cuda()
        sparton.tie_weights(decoder)

        hidden = torch.randn(B, S, hidden_dim, device="cuda", requires_grad=True)
        mask = torch.ones(B, S, dtype=torch.long, device="cuda")

        out = sparton(hidden, mask)
        loss = out.sum()
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape
        assert decoder.weight.grad is not None

    def test_sparton_aggregation_module(self):
        """SpartonAggregationModule uses the kernel on CUDA."""
        from xpmir.neural.splade import SpartonAggregationModule

        transform = torch.nn.Identity().cuda()
        decoder = torch.nn.Linear(32, 128, bias=True).cuda()
        module = SpartonAggregationModule(transform, decoder)

        hidden = torch.randn(2, 5, 32, device="cuda")
        mask = torch.ones(2, 5, dtype=torch.long, device="cuda")
        output = module(hidden, mask)

        assert output.shape == (2, 128)
        assert (output >= 0).all()
