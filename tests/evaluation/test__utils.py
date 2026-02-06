"""Tests for callcut.evaluation._utils module."""

from __future__ import annotations

import pytest
from numpy.testing import assert_allclose

from callcut.evaluation._utils import _precision_recall_f1


class TestPrecisionRecallF1:
    """Tests for _precision_recall_f1."""

    def test_perfect_scores(self) -> None:
        """Test perfect precision, recall, and F1."""
        precision, recall, f1 = _precision_recall_f1(tp=10, fp=0, fn=0)
        assert_allclose(precision, 1.0, atol=1e-6)
        assert_allclose(recall, 1.0, atol=1e-6)
        assert_allclose(f1, 1.0, atol=1e-6)

    def test_zero_tp(self) -> None:
        """Test all zeros when no true positives."""
        precision, recall, f1 = _precision_recall_f1(tp=0, fp=5, fn=3)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_no_false_positives(self) -> None:
        """Test perfect precision with some false negatives."""
        precision, recall, f1 = _precision_recall_f1(tp=7, fp=0, fn=3)
        assert_allclose(precision, 1.0, atol=1e-6)
        assert_allclose(recall, 0.7, atol=1e-6)
        assert f1 > 0.0

    def test_no_false_negatives(self) -> None:
        """Test perfect recall with some false positives."""
        precision, recall, f1 = _precision_recall_f1(tp=7, fp=3, fn=0)
        assert_allclose(recall, 1.0, atol=1e-6)
        assert_allclose(precision, 0.7, atol=1e-6)
        assert f1 > 0.0

    def test_all_zeros(self) -> None:
        """Test all zeros when everything is zero."""
        precision, recall, f1 = _precision_recall_f1(tp=0, fp=0, fn=0)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_known_values(self) -> None:
        """Test against hand-computed values."""
        # tp=6, fp=2, fn=4 -> precision=6/8=0.75, recall=6/10=0.6
        # f1 = 2*0.75*0.6/(0.75+0.6) = 0.9/1.35 = 2/3
        precision, recall, f1 = _precision_recall_f1(tp=6, fp=2, fn=4)
        assert_allclose(precision, 0.75, atol=1e-6)
        assert_allclose(recall, 0.6, atol=1e-6)
        assert_allclose(f1, 2.0 / 3.0, atol=1e-6)

    @pytest.mark.parametrize(
        ("tp", "fp", "fn"),
        [(1, 0, 0), (10, 5, 5), (100, 100, 0), (0, 100, 100)],
    )
    def test_f1_between_zero_and_one(self, tp: int, fp: int, fn: int) -> None:
        """Test F1 is always in [0, 1]."""
        _, _, f1 = _precision_recall_f1(tp, fp, fn)
        assert 0.0 <= f1 <= 1.0
