import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def predictive_entropy(probs: np.ndarray) -> float:
    """Shannon entropy."""
    eps = 1e-12
    return float(-np.sum(probs * np.log(probs + eps)))


def margin_top2(probs: np.ndarray) -> float:
    """Difference between top-1 and top-2 probabilities."""
    top2 = np.sort(probs)[-2:]
    return float(top2[1] - top2[0])


def top2_ratio(probs: np.ndarray) -> float:
    """Ratio between top-1 and top-2 probabilities."""
    top2 = np.sort(probs)[-2:]
    eps = 1e-12
    return float(top2[1] / (top2[0] + eps))


def uncertainty_metrics(logits: np.ndarray, temperature: float = 1.0) -> dict:
    """Compute all uncertainty metrics from logits."""
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)

    top_indices = np.argsort(probs)[-2:][::-1]

    return {
        "predicted_class": int(top_indices[0]),
        "confidence": float(probs[top_indices[0]]),
        "top2_classes": top_indices.tolist(),
        "top2_probs": probs[top_indices].tolist(),
        "entropy": predictive_entropy(probs),
        "margin": margin_top2(probs),
        "top2_ratio": top2_ratio(probs),
        "full_probs": probs.tolist(),
    }
