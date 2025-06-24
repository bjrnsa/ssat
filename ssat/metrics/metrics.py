# metrics.py.
import numpy as np


def multiclass_brier_score(y_true, y_prob):
    """Compute the multiclass Brier score.

    Parameters:
        y_true: (n_samples,) array of true class indices
        y_prob: (n_samples, n_classes) array of predicted probabilities

    Returns:
        float: mean Brier score
    """
    # Convert y_true to one-hot encoding
    n_samples, n_classes = y_prob.shape
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1.0

    # Compute squared differences and return mean
    diff = y_prob - y_true_onehot
    return np.mean(diff**2)


def ignorance_score(y_true, y_prob):
    """Compute the mean Ignorance Score for a set of probabilistic predictions.

    Parameters:
        y_true (1D np.ndarray[int]): Actual match results (0 = Home, 1 = Draw, 2 = Away).
        y_prob (2D np.ndarray[float64]): Predicted probability distributions,
                                         shape (n_samples, 3).

    Returns:
        float: Mean ignorance score.
    """
    epsilon = 1e-15

    # Extract probabilities for true classes
    true_class_probs = y_prob[np.arange(len(y_true)), y_true]

    # Clip probabilities to avoid log(0)
    true_class_probs = np.clip(true_class_probs, epsilon, 1.0)

    # Compute negative log2 and return mean
    return -np.mean(np.log2(true_class_probs))


def rps_array(y_true, y_prob):
    """Compute individual RPS scores for each fixture and return them as an array.

    Parameters:
        outcomes: Array (length n_samples) of observed outcome indices
        probs: 2D probability array of shape (n_samples, n_outcomes)

    Returns:
        np.ndarray: RPS scores for each fixture
    """
    n_samples, n_outcomes = y_prob.shape

    # Validate outcomes
    if np.any((y_true < 0) | (y_true >= n_outcomes)):
        raise ValueError("Invalid outcome index")

    # Compute cumulative probabilities
    cum_probs = np.cumsum(y_prob, axis=1)

    # Create indicator matrix (one-hot encoding)
    indicator = np.zeros_like(y_prob)
    indicator[np.arange(n_samples), y_true] = 1.0

    # Compute cumulative outcomes
    cum_outcomes = np.cumsum(indicator, axis=1)

    # Compute squared differences
    diff = cum_probs - cum_outcomes
    squared_diff = diff**2

    # Sum over outcomes and normalize
    rps_scores = np.sum(squared_diff, axis=1) / (n_outcomes - 1.0)

    return rps_scores


def average_rps(y_true, y_prob):
    """Compute the average RPS over all fixtures.

    Parameters:
        outcomes: Array (length n_samples) of observed outcome indices
        probs: 2D probability array of shape (n_samples, n_outcomes)

    Returns:
        float: The average RPS
    """
    rps_scores = rps_array(y_true, y_prob)
    return np.mean(rps_scores)


def binary_brier_score(y_true, y_prob):
    """Compute the binary Brier score (special case of multiclass).

    Parameters:
        y_true: (n_samples,) array of true binary labels (0 or 1)
        y_prob: (n_samples,) array of predicted probabilities for class 1

    Returns:
        float: mean Brier score
    """
    return np.mean((y_prob - y_true) ** 2)


def multiclass_log_loss(y_true, y_prob):
    """Compute the multiclass log loss (cross-entropy).

    Parameters:
        y_true: (n_samples,) array of true class indices
        y_prob: (n_samples, n_classes) array of predicted probabilities

    Returns:
        float: mean log loss
    """
    epsilon = 1e-15

    # Extract probabilities for true classes
    true_class_probs = y_prob[np.arange(len(y_true)), y_true]

    # Clip probabilities to avoid log(0)
    true_class_probs = np.clip(true_class_probs, epsilon, 1.0)

    # Compute negative log and return mean
    return -np.mean(np.log(true_class_probs))


def calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE).

    Parameters:
        y_true: (n_samples,) array of true class indices
        y_prob: (n_samples, n_classes) array of predicted probabilities
        n_bins: number of bins for calibration

    Returns:
        float: Expected Calibration Error
    """
    # Get max probabilities and predictions
    max_probs = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)

    # Check if predictions are correct
    correct = (predictions == y_true).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Compute accuracy and confidence in this bin
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()

            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def balanced_accuracy(y_true, y_prob):
    """Compute balanced accuracy (macro-averaged recall).

    Better for imbalanced datasets where some classes are rarely predicted.

    Parameters:
        y_true: (n_samples,) array of true class indices
        y_prob: (n_samples, n_classes) array of predicted probabilities

    Returns:
        float: balanced accuracy (average of per-class recalls)
    """
    y_pred = np.argmax(y_prob, axis=1)
    n_classes = y_prob.shape[1]

    class_accuracies = []
    for class_idx in range(n_classes):
        # Get samples that truly belong to this class
        class_mask = y_true == class_idx
        if np.sum(class_mask) > 0:  # Only if class exists in true labels
            # Calculate recall for this class
            class_recall = np.mean(y_pred[class_mask] == class_idx)
            class_accuracies.append(class_recall)

    return np.mean(class_accuracies) if class_accuracies else 0.0
