import numpy as np
import pytest

from churn.eval.economics import break_even, net_savings_curve, shift_base_rate

ECON = dict(contact_cost=50.0, success_rate=0.2, customer_value=1000.0)  # break-even p = 0.25


def test_curve_counts_saved_value_minus_contact_cost():
    score = np.array([0.9, 0.1, 0.5])
    churned = np.array([1, 0, 0])
    # order: 0.9 (churner, +200-50), 0.5 (stays, -50), 0.1 (stays, -50)
    np.testing.assert_allclose(net_savings_curve(score, churned, **ECON), [0, 150, 100, 50])


def test_curve_peaks_where_probability_crosses_break_even():
    p = np.linspace(0.6, 0.0, 61)
    curve = net_savings_curve(p, p, **ECON)
    k = int(np.argmax(curve))
    assert break_even(**ECON) == 0.25
    assert p[k - 1] >= 0.25 > p[k]  # last contacted user is at/above break-even, next is below


def test_contacting_nobody_is_an_option():
    curve = net_savings_curve(np.array([0.1, 0.05]), np.array([0.1, 0.05]), **ECON)
    assert int(np.argmax(curve)) == 0


@pytest.mark.parametrize("target", [0.02, 0.08])
def test_base_rate_shift_hits_target_and_keeps_order(target):
    p = np.random.default_rng(0).beta(1, 20, 1000)
    shifted = shift_base_rate(p, target)
    assert shifted.mean() == pytest.approx(target, rel=1e-6)
    np.testing.assert_array_equal(np.argsort(p, kind="stable"), np.argsort(shifted, kind="stable"))
