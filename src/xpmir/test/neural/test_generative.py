from xpmir.neural.generative.hf import T5ConditionalGenerator, LoadFromT5
from xpmir.neural.generative import BeamSearchGenerationOptions
from xpmir.test import skip_if_ci
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


@skip_if_ci
@torch.no_grad()
def test_t5_generate():
    """Test consistency in generation"""

    hf_id = "t5-small"
    # our model
    options = BeamSearchGenerationOptions(num_return_sequences=5, num_beams=5)
    test_text = ["test one", "test two"]
    our_model: T5ConditionalGenerator = T5ConditionalGenerator(
        hf_id=hf_id,
    ).instance()
    LoadFromT5(t5_model=our_model).execute()
    our_model.eval()
    our_output = our_model.generate(test_text, options)
    our_sequence = our_output.sequences

    # official model
    tokenizer = T5Tokenizer.from_pretrained(hf_id)
    official_model = T5ForConditionalGeneration.from_pretrained(hf_id)
    official_model.eval()

    input = tokenizer(
        test_text,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    official_output = official_model.generate(
        input_ids=input["input_ids"],
        attention_mask=input["attention_mask"],
        max_new_tokens=10,
        num_beams=5,
        num_return_sequences=5,
        return_dict_in_generate=True,
        output_scores=True,
    )
    official_sequences = official_output.sequences

    assert torch.equal(our_sequence, official_sequences)
