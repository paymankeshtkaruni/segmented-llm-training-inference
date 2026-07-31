"""Training utilities for segmented LLM training."""

try:
    from sequential_segmented_llm_training_inference.training.losses import (
        CausalCrossEntropyConfig,
        CausalCrossEntropyLoss,
        causal_cross_entropy_loss,
        shift_logits_and_labels,
        validate_causal_lm_shapes,
    )
except ImportError:  # pragma: no cover - allows phased installation
    CausalCrossEntropyConfig = None  # type: ignore[assignment]
    CausalCrossEntropyLoss = None  # type: ignore[assignment]
    causal_cross_entropy_loss = None  # type: ignore[assignment]
    shift_logits_and_labels = None  # type: ignore[assignment]
    validate_causal_lm_shapes = None  # type: ignore[assignment]

try:
    from sequential_segmented_llm_training_inference.training.validator import (
        SegmentedValidator,
        ValidationResult,
    )
except ImportError:  # pragma: no cover - allows phased installation
    SegmentedValidator = None  # type: ignore[assignment]
    ValidationResult = None  # type: ignore[assignment]

from sequential_segmented_llm_training_inference.training.tester import (
    SegmentedTester,
    TestResult,
)

__all__ = [
    "CausalCrossEntropyConfig",
    "CausalCrossEntropyLoss",
    "causal_cross_entropy_loss",
    "shift_logits_and_labels",
    "validate_causal_lm_shapes",
    "SegmentedValidator",
    "ValidationResult",
    "SegmentedTester",
    "TestResult",
]
