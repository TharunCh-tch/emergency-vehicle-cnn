"""Tests for metrics computation: a fake confusion matrix -> correct
precision/recall/F1 math, checked by hand."""

from src.metrics import compute_metrics


def test_perfect_predictions():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    m = compute_metrics(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.confusion_matrix == [[2, 0], [0, 3]]


def test_known_confusion_matrix_values():
    # 10 samples: TN=3, FP=2, FN=1, TP=4
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    m = compute_metrics(y_true, y_pred)

    tn, fp, fn, tp = 3, 2, 1, 4
    expected_precision = tp / (tp + fp)  # 4/6
    expected_recall = tp / (tp + fn)  # 4/5
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    expected_accuracy = (tp + tn) / 10

    assert m.confusion_matrix == [[tn, fp], [fn, tp]]
    assert abs(m.accuracy - expected_accuracy) < 1e-9
    assert abs(m.precision - expected_precision) < 1e-9
    assert abs(m.recall - expected_recall) < 1e-9
    assert abs(m.f1 - expected_f1) < 1e-9


def test_all_wrong_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert m.accuracy == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.confusion_matrix == [[0, 2], [2, 0]]


def test_no_positive_predictions_precision_zero_division_handled():
    y_true = [0, 1, 1]
    y_pred = [0, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert m.precision == 0.0  # zero_division=0, no crash
    assert m.recall == 0.0


def test_roc_auc_computed_when_scores_given():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]  # classic textbook example, AUC = 0.75
    m = compute_metrics(y_true, y_pred=[0, 0, 0, 1], y_score=y_score)
    assert m.roc_auc is not None
    assert abs(m.roc_auc - 0.75) < 1e-9


def test_roc_auc_none_when_single_class_present():
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    y_score = [0.9, 0.8, 0.95]
    m = compute_metrics(y_true, y_pred, y_score)
    assert m.roc_auc is None


def test_to_dict_contains_all_keys():
    m = compute_metrics([0, 1], [0, 1])
    d = m.to_dict()
    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"):
        assert key in d
