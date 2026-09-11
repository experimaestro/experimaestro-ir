import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


def wrap_collate_with_tokenizer(
    model: Any,
    collate_fn: Callable[[List[Any]], Any],
    *,
    records_key: str = "records",
    tokenized_records_key: str = "tokenized_records",
    log: Optional[logging.Logger] = None,
) -> Callable[[List[Any]], Any]:
    """Wrap a DataLoader collate function to pre-tokenize records on CPU workers.

    If `model` (or `model.module`) implements `get_tokenizer_fn()`, the returned
    tokenizer callable is used by worker processes to tokenize `inputs[records_key]`
    and store the result into `inputs[tokenized_records_key]`.

    If `get_tokenizer_fn()` is missing, or if it returns None / non-callable, a warning
    is emitted and the original `collate_fn` is returned unchanged.

    Args:
        model: The model or scorer instance.
        collate_fn: The base batch collate function.
        records_key: The dict key containing the batch records (default: "records").
        tokenized_records_key: The dict key under which tokenized records are stored
            (default: "tokenized_records").
        log: Custom logger to use for warnings (defaults to module logger).

    Returns:
        A wrapped collate function with tokenization, or the original collate function.
    """
    _logger = log or logger
    unwrapped_model = getattr(model, "module", model)
    model_name = type(unwrapped_model).__name__

    if not hasattr(unwrapped_model, "get_tokenizer_fn"):
        _logger.warning(
            "Model %s does not implement `get_tokenizer_fn()`. "
            "Inputs will not be pre-tokenized on CPU workers during data loading.",
            model_name,
        )
        return collate_fn

    tokenization_fn = unwrapped_model.get_tokenizer_fn()
    if tokenization_fn is None or not callable(tokenization_fn):
        _logger.warning(
            "Model %s implements `get_tokenizer_fn()`, but failed to grab a callable preprocessing function (got %r). "
            "Inputs will not be pre-tokenized on CPU workers during data loading.",
            model_name,
            tokenization_fn,
        )
        return collate_fn

    def collate_fn_with_tokenization(samples: List[Any]) -> Any:
        inputs = collate_fn(samples)
        inputs[tokenized_records_key] = tokenization_fn(inputs[records_key])
        return inputs

    return collate_fn_with_tokenization
