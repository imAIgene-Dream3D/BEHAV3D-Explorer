"""Automatic node ordering of the circular transition diagram: crossings and overlap are measured
on the arrows as they are drawn, the search finds the true minimum, and both reports use one order
on every circular page without changing any saved probability. Also covers zero-probability
transitions, which are never drawn.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/behav3d-mpl-cache")

from itertools import permutations
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionStyle, FancyArrowPatch

from behav3d.analysis.behavior.general.visualization.plots import circular_transition_diagram as ctd
from behav3d.analysis.behavior.state.visualization.plots import state_transitions
from behav3d.analysis.behavior.track.visualization.plots import transition_analysis

# Saved order "abcdef" visits the states so that arrows cross; going round this ring they don't.
_RING = list("adbecf")
_SAVED_ORDER = list("abcdef")
_APP = None
_PLOT_FUNCTIONS = (
    "plot_circular_transition_diagram",
    "plot_circular_transition_diagram_pair",
    "plot_circular_transition_diagram_grid",
)


def _random_probs(n, seed, density=0.6):
    """Row-normalized transition probabilities with an empty diagonal and some zero entries."""
    rng = np.random.default_rng(seed)
    values = rng.random((n, n)) * (rng.random((n, n)) < density)
    np.fill_diagonal(values, 0.0)
    labels = [f"s{i}" for i in range(n)]
    probs = pd.DataFrame(values, index=labels, columns=labels)
    return probs.div(probs.sum(axis=1).replace(0, np.nan), axis=0)


def _drawn_arrow_count(ax):
    return sum(isinstance(patch, FancyArrowPatch) for patch in ax.patches)


def _capture_plot_orders(monkeypatch, module):
    """Record the `state_order` of every circular page a report draws."""
    orders = []
    for name in _PLOT_FUNCTIONS:
        real = getattr(module, name)

        def recording(*args, _real=real, **kwargs):
            orders.append(list(kwargs["state_order"]))
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, recording)
    return orders


def test_arrow_curve_matches_matplotlib_arc3():
    positions = ctd._circular_node_positions(list(range(7)))
    for src, dst in [(0, 1), (0, 3), (2, 6), (5, 1)]:
        start, end = ctd._arrow_endpoints(positions[src], positions[dst])
        path = ConnectionStyle.Arc3(rad=0.28)(start, end, shrinkA=0, shrinkB=0)
        p0, control, p1 = path.vertices[:3]
        t = np.linspace(0, 1, ctd._CROSSING_CURVE_SEGMENTS + 1)[:, None]
        expected = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * control + t ** 2 * p1
        assert np.allclose(ctd._arrow_curve(positions[src], positions[dst], 0.28), expected)


def test_arrow_endpoints_match_the_drawn_arrows():
    labels = list("abcd")
    probs = pd.DataFrame(0.0, index=labels, columns=labels)
    probs.loc["a", "c"] = 1.0
    fig, ax = plt.subplots()
    ctd._draw_circular_transition_diagram_on_ax(
        ax, probs, colors={label: "#000000" for label in labels}, min_prob_to_draw=0.0,
    )
    arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    positions = ctd._circular_node_positions(labels)
    assert len(arrows) == 1
    assert np.allclose(arrows[0]._posA_posB, ctd._arrow_endpoints(positions["a"], positions["c"]))
    plt.close(fig)


def test_arrow_pair_tables_match_direct_geometry():
    n = 7
    crossings, overlap = ctd._arrow_pair_tables(n, 0.28)
    positions = ctd._circular_node_positions(list(range(n)))
    arrows = [(src, dst) for src in range(n) for dst in range(n) if src != dst]
    curves = {arrow: ctd._arrow_curve(positions[arrow[0]], positions[arrow[1]], 0.28) for arrow in arrows}
    for i, a in enumerate(arrows):
        for b in arrows[i + 1:]:
            ids = (a[0] * n + a[1], b[0] * n + b[1])
            assert crossings[ids] == ctd._count_curve_crossings(curves[a], curves[b]), (a, b)
            assert overlap[ids] == pytest.approx(ctd._curve_overlap(curves[a], curves[b]), abs=0.05), (a, b)


def _clutter(probs, order, min_prob=0.0):
    n = len(order)
    arrows = ctd._drawn_arrows(probs, order, min_prob)
    return int(ctd._pair_totals_per_order(np.arange(n)[None, :], arrows, ctd._clutter_table(n, 0.28))[0])


@pytest.mark.parametrize("min_prob", [0.0, 0.2])
def test_search_finds_the_brute_force_minimum(min_prob):
    outgoing = _random_probs(6, seed=1)
    incoming = _random_probs(6, seed=2)
    labels = list(outgoing.index)

    def score(order):
        return (
            _clutter(outgoing, order, min_prob) + _clutter(incoming, order, min_prob),
            _clutter(outgoing, order) + _clutter(incoming, order),
        )

    best = min(score([labels[0], *rest]) for rest in permutations(labels[1:]))
    assert best[1] > 0  # otherwise this test could not tell a good order from a bad one
    assert score(ctd.least_cluttered_order([outgoing, incoming], min_prob)) == best


def test_search_avoids_stacking_arrows_to_save_a_few_crossings():
    # The arrows of a real incoming diagram, in the order a search counting only crossings chose:
    # few crossings, but many arrows running on top of each other.
    crossings_only_order = list("abcdefgh")
    arrows = [
        "fe", "fg", "fh", "fa", "gf", "gd", "gh", "ga", "de",
        "dc", "db", "dg", "ed", "eb", "cd", "hg", "hb", "ha",
    ]
    probs = pd.DataFrame(0.0, index=crossings_only_order, columns=crossings_only_order)
    for src, dst in arrows:
        probs.loc[src, dst] = 0.5

    order = ctd.least_cluttered_order([probs])
    assert ctd.overlap_score(probs, order) < ctd.overlap_score(probs, crossings_only_order) / 1.5
    assert ctd.count_crossings(probs, order) <= ctd.count_crossings(probs, crossings_only_order) + 2


def test_overlap_score_only_counts_arrows_running_close_together():
    labels = list("abcdef")
    empty = pd.DataFrame(0.0, index=labels, columns=labels)
    assert ctd.overlap_score(empty, labels) == 0.0

    # Two short arrows on opposite sides of the circle stay apart...
    apart = empty.copy()
    apart.loc["a", "b"] = apart.loc["d", "e"] = 1.0
    assert ctd.overlap_score(apart, labels) == 0.0

    # ...while the two directions between two neighbouring nodes run close together.
    close = empty.copy()
    close.loc["a", "b"] = close.loc["b", "a"] = 1.0
    assert ctd.overlap_score(close, labels) > 0.0


def test_crossings_do_not_change_when_the_diagram_is_rotated():
    probs = _random_probs(8, seed=3)
    order = list(probs.index)
    expected = ctd.count_crossings(probs, order)
    assert expected > 0
    for shift in range(1, 8):
        assert ctd.count_crossings(probs, order[shift:] + order[:shift]) == expected


def test_a_tie_keeps_the_existing_order():
    labels = list("dbac")
    empty = pd.DataFrame(0.0, index=labels, columns=labels)
    assert ctd.least_cluttered_order([empty]) == labels

    one_arrow = empty.copy()
    one_arrow.loc["d", "b"] = 1.0
    assert ctd.least_cluttered_order([one_arrow]) == labels


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_very_few_states(n):
    probs = _random_probs(n, seed=4) if n else pd.DataFrame()
    order = ctd.least_cluttered_order([probs])
    assert sorted(order) == sorted(probs.index)
    assert ctd.count_crossings(probs, order) == 0


def test_rows_without_any_transition_are_ignored():
    probs = _random_probs(5, seed=5)
    probs.iloc[2] = np.nan
    order = ctd.least_cluttered_order([probs])
    assert sorted(order) == sorted(probs.index)


def test_non_string_labels_come_back_as_the_labels_the_plot_uses():
    probs = _random_probs(5, seed=6)
    probs.index = probs.columns = range(5)
    order = ctd.least_cluttered_order([probs])
    assert sorted(order) == list("01234")
    assert ctd.count_crossings(probs, order) == ctd.count_crossings(
        probs.rename(index=str, columns=str), order,
    )


def test_swap_search_is_repeatable_and_never_worse_than_the_saved_order():
    probs = _random_probs(12, seed=7)
    order = ctd.least_cluttered_order([probs])
    assert sorted(order) == sorted(probs.index)
    assert ctd.least_cluttered_order([probs]) == order
    assert _clutter(probs, order) < _clutter(probs, list(probs.index))


def test_zero_probability_transitions_are_not_drawn():
    probs = _random_probs(5, seed=8)
    fig, ax = plt.subplots()
    ctd._draw_circular_transition_diagram_on_ax(
        ax, probs, colors={label: "#000000" for label in probs.index}, min_prob_to_draw=0.0,
    )
    assert _drawn_arrow_count(ax) == int((probs > 0).sum().sum())
    plt.close(fig)


def test_self_transitions_above_zero_are_still_drawn_as_loops():
    labels = list("abc")
    probs = pd.DataFrame(0.0, index=labels, columns=labels)
    probs.loc["a", "a"] = 0.5
    probs.loc["a", "b"] = 0.5
    fig, ax = plt.subplots()
    ctd._draw_circular_transition_diagram_on_ax(
        ax, probs, colors={label: "#000000" for label in labels}, min_prob_to_draw=0.0,
    )
    assert _drawn_arrow_count(ax) == 2
    plt.close(fig)


def _state_adata():
    states = _RING * 5
    obs = pd.DataFrame(
        {
            "sample_name": ["s1"] * len(states),
            "TrackID": [1] * len(states),
            "position_t": list(range(len(states))),
            "full_behavioral_state": states,
        }
    )
    obs.index = obs.index.astype(str)
    return ad.AnnData(obs=obs)


def _run_state_report(tmp_path, **kwargs):
    return state_transitions.save_state_transition_report(
        adata=_state_adata(),
        output_dir=tmp_path,
        state_col="full_behavioral_state",
        state_order=_SAVED_ORDER,
        include_circular_absolute_scale=True,
        include_ngram_rankings=False,
        verbose=False,
        **kwargs,
    )


def _no_self_probs(out):
    return pd.read_csv(out["transition_matrix_probs_no_self_csv"], index_col=0)


def test_state_report_uses_one_arranged_order_on_every_circular_page(tmp_path, monkeypatch):
    orders = _capture_plot_orders(monkeypatch, state_transitions)
    out = _run_state_report(tmp_path)

    order = out["circular_node_order"]
    assert len(orders) == 6
    assert all(page_order == order for page_order in orders)
    assert order != _SAVED_ORDER

    probs = _no_self_probs(out)
    assert ctd.count_crossings(probs, _SAVED_ORDER) > 0
    assert ctd.count_crossings(probs, order) == 0


def test_state_report_turned_off_keeps_the_saved_order(tmp_path, monkeypatch):
    orders = _capture_plot_orders(monkeypatch, state_transitions)
    out = _run_state_report(tmp_path, circular_auto_node_order=False)

    assert out["circular_node_order"] == _SAVED_ORDER
    assert all(page_order == _SAVED_ORDER for page_order in orders)


def test_state_report_saved_data_does_not_depend_on_the_node_order(tmp_path):
    arranged = _run_state_report(tmp_path / "on", circular_auto_node_order=True)
    saved = _run_state_report(tmp_path / "off", circular_auto_node_order=False)
    for key in ("transition_matrix_counts_csv", "transition_matrix_probs_csv", "transition_matrix_probs_no_self_csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(arranged[key], index_col=0), pd.read_csv(saved[key], index_col=0),
        )
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "on" / "transition_matrix" / "data" / "transition_matrix_probs_incoming.csv", index_col=0),
        pd.read_csv(tmp_path / "off" / "transition_matrix" / "data" / "transition_matrix_probs_incoming.csv", index_col=0),
    )
    assert list(pd.read_csv(arranged["transition_matrix_probs_csv"], index_col=0).index) == _SAVED_ORDER


def test_state_report_without_circular_pages_has_no_node_order(tmp_path):
    out = _run_state_report(tmp_path, include_circular_diagram=False)
    assert out["circular_node_order"] is None


def _track_adata():
    clusters = _RING * 5
    obs = pd.DataFrame(
        {
            "sample_name": ["s1"] * len(clusters),
            "TrackID": [1] * len(clusters),
            "trajectory_window_id": list(range(len(clusters))),
            "ClusterID": clusters,
        }
    )
    obs.index = obs.index.astype(str)
    return ad.AnnData(obs=obs)


def _run_track_report(tmp_path, **kwargs):
    return transition_analysis.save_window_cluster_transition_analysis(
        _track_adata(),
        output_dir=str(tmp_path),
        cell_type="tcell",
        cluster_key="ClusterID",
        state_order=_SAVED_ORDER,
        include_circular_absolute_scale=True,
        verbose=False,
        **kwargs,
    )


def test_track_report_uses_one_arranged_order_on_every_circular_page(tmp_path, monkeypatch):
    orders = _capture_plot_orders(monkeypatch, transition_analysis)
    out = _run_track_report(tmp_path)

    order = out["circular_node_order"]
    assert len(orders) == 6
    assert all(page_order == order for page_order in orders)

    probs = pd.read_csv(out["transition_matrix_probs_no_self_csv"], index_col=0)
    assert ctd.count_crossings(probs, _SAVED_ORDER) > 0
    assert ctd.count_crossings(probs, order) == 0


def test_track_report_turned_off_keeps_the_saved_order(tmp_path, monkeypatch):
    orders = _capture_plot_orders(monkeypatch, transition_analysis)
    out = _run_track_report(tmp_path, auto_node_order=False)

    assert out["circular_node_order"] == _SAVED_ORDER
    assert all(page_order == _SAVED_ORDER for page_order in orders)


def _capture_titled_matrices(monkeypatch, module):
    """Record the title and matrix of every circular page a report draws."""
    pages = []
    for name in _PLOT_FUNCTIONS:
        real = getattr(module, name)

        def recording(probs, *args, _real=real, **kwargs):
            pages.append((kwargs["title"], probs))
            return _real(probs, *args, **kwargs)

        monkeypatch.setattr(module, name, recording)
    return pages


def _assert_pages_are_labeled_by_their_data(pages):
    """Outgoing pages must show row-normalized data, incoming pages column-normalized data."""
    assert len(pages) == 6
    for title, probs in pages:
        assert ("Outgoing" in title) != ("Incoming" in title), title
        axis = 1 if "Outgoing" in title else 0
        assert np.allclose(probs.sum(axis=axis), 1.0), title
    assert sum("Outgoing" in title for title, _ in pages) == 3


def test_state_report_pages_are_labeled_outgoing_or_incoming(tmp_path, monkeypatch):
    pages = _capture_titled_matrices(monkeypatch, state_transitions)
    _run_state_report(tmp_path)
    _assert_pages_are_labeled_by_their_data(pages)


def test_track_report_pages_are_labeled_outgoing_or_incoming(tmp_path, monkeypatch):
    pages = _capture_titled_matrices(monkeypatch, transition_analysis)
    _run_track_report(tmp_path)
    _assert_pages_are_labeled_by_their_data(pages)


def _qt_app():
    global _APP
    from qtpy.QtWidgets import QApplication

    import behav3d.napari._analysis  # noqa: F401  (resolves a circular import first)

    _APP = QApplication.instance() or QApplication([])


def _record_results(monkeypatch, module, name):
    """Record what a report returns when the app runs it."""
    results = []
    real = getattr(module, name)

    def recording(*args, **kwargs):
        results.append(real(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(module, name, recording)
    return results


def _run_now(tab):
    """Run the tab's background job straight away, on this thread."""
    tab._bg = SimpleNamespace(
        is_running=lambda: False,
        run=lambda fn, on_done, **kwargs: on_done(fn()),
    )


@pytest.mark.parametrize("ticked", [True, False])
def test_state_tab_checkbox_reaches_the_report(tmp_path, monkeypatch, ticked):
    _qt_app()
    from behav3d.napari._single_cell import StateClassificationSubTab

    h5ad_path = tmp_path / "states.h5ad"
    _state_adata().write_h5ad(h5ad_path)
    results = _record_results(monkeypatch, state_transitions, "save_state_transition_report")

    loader = SimpleNamespace(output_dir=str(tmp_path), behav3d_parameters={}, metadata=None)
    tab = StateClassificationSubTab(viewer=None, metadata_loader=loader, cell_type_getter=lambda: "tcells")
    _run_now(tab)
    tab._full_adata_path = lambda cell_type: h5ad_path
    tab.chk_state_transition_auto_order.setChecked(ticked)
    tab.chk_state_transition_include_ngrams.setChecked(False)
    tab._on_state_transition()

    assert len(results) == 1
    assert (results[0]["circular_node_order"] != _SAVED_ORDER) == ticked


@pytest.mark.parametrize("ticked", [True, False])
def test_track_tab_checkbox_reaches_the_report(tmp_path, monkeypatch, ticked):
    _qt_app()
    from behav3d.napari._single_cell import TrackClassificationSubTab

    results = _record_results(monkeypatch, transition_analysis, "save_window_cluster_transition_analysis")

    loader = SimpleNamespace(output_dir=str(tmp_path), behav3d_parameters={}, metadata=None)
    tab = TrackClassificationSubTab(viewer=None, metadata_loader=loader, cell_type_getter=lambda: "tcells")
    _run_now(tab)
    tab._track_adata = _track_adata()
    tab.chk_transanalysis_auto_order.setChecked(ticked)
    tab.chk_transanalysis_include_window_sankey.setChecked(False)
    tab._on_transition_analysis()

    assert len(results) == 1
    assert (results[0]["circular_node_order"] != _SAVED_ORDER) == ticked


_CUSTOM_ORDER = list("fedcba")


def test_state_report_uses_a_custom_order_on_every_circular_page(tmp_path, monkeypatch):
    orders = _capture_plot_orders(monkeypatch, state_transitions)
    out = _run_state_report(tmp_path, circular_auto_node_order=False, circular_custom_node_order=_CUSTOM_ORDER)

    assert out["circular_node_order"] == _CUSTOM_ORDER
    assert len(orders) == 6
    assert all(page_order == _CUSTOM_ORDER for page_order in orders)
    assert list(_no_self_probs(out).index) == _SAVED_ORDER


def test_track_report_uses_a_custom_order_on_every_circular_page(tmp_path, monkeypatch):
    orders = _capture_plot_orders(monkeypatch, transition_analysis)
    out = _run_track_report(tmp_path, auto_node_order=False, custom_node_order=_CUSTOM_ORDER)

    assert out["circular_node_order"] == _CUSTOM_ORDER
    assert all(page_order == _CUSTOM_ORDER for page_order in orders)


def test_custom_order_with_missing_and_unknown_states_keeps_every_node(tmp_path):
    out = _run_state_report(tmp_path, circular_auto_node_order=False, circular_custom_node_order=["c", "gone", "a"])
    assert out["circular_node_order"] == ["c", "a", "b", "d", "e", "f"]


def test_auto_arranging_takes_priority_over_a_custom_order(tmp_path):
    out = _run_state_report(tmp_path, circular_auto_node_order=True, circular_custom_node_order=_CUSTOM_ORDER)
    assert ctd.count_crossings(_no_self_probs(out), out["circular_node_order"]) == 0
    assert out["circular_node_order"] != _CUSTOM_ORDER


def _node_order_list():
    _qt_app()
    from behav3d.napari._single_cell import NodeOrderList

    return NodeOrderList("states")


def test_node_order_list_starts_in_the_saved_order_and_keeps_user_changes_on_refresh():
    widget = _node_order_list()
    widget.set_labels(list("abc"))
    assert widget.order() == list("abc")

    widget.set_order(list("cab"))
    widget.set_labels(list("abc"))
    assert widget.order() == list("cab")

    # A state that disappears is dropped and a new one goes last.
    widget.set_labels(list("abd"))
    assert widget.order() == list("abd")
    widget.set_order(list("dab"))
    widget.set_labels(list("abde"))
    assert widget.order() == list("dabe")


def test_node_order_list_buttons():
    widget = _node_order_list()
    widget.set_labels(list("abc"))
    widget.set_order(list("cba"))
    widget.btn_reset.click()
    assert widget.order() == list("abc")

    assert not widget.btn_use_auto.isEnabled()
    widget.set_last_auto_order(list("bca"))
    assert widget.btn_use_auto.isEnabled()
    widget.btn_use_auto.click()
    assert widget.order() == list("bca")


def test_node_order_list_is_only_active_while_not_auto_arranging():
    widget = _node_order_list()
    widget.set_labels(list("abc"))
    widget.set_last_auto_order(list("bca"))
    widget.set_auto_arranging(True)
    assert not widget.list_widget.isEnabled()
    assert not widget.btn_reset.isEnabled()
    assert not widget.btn_use_auto.isEnabled()

    # A report finishing while auto-arranging must not wake the button up.
    widget.set_last_auto_order(list("cab"))
    assert not widget.btn_use_auto.isEnabled()

    widget.set_auto_arranging(False)
    assert widget.list_widget.isEnabled()
    assert widget.btn_use_auto.isEnabled()


def _state_tab_with_data(tmp_path):
    _qt_app()
    from behav3d.napari._single_cell import StateClassificationSubTab

    h5ad_path = tmp_path / "states.h5ad"
    adata = _state_adata()
    adata.write_h5ad(h5ad_path)
    loader = SimpleNamespace(output_dir=str(tmp_path), behav3d_parameters={}, metadata=None)
    tab = StateClassificationSubTab(viewer=None, metadata_loader=loader, cell_type_getter=lambda: "tcells")
    _run_now(tab)
    tab._full_adata_path = lambda cell_type: h5ad_path
    tab.chk_state_transition_include_ngrams.setChecked(False)
    tab._refresh_transition_node_order(adata, "full_behavioral_state")
    return tab


def test_state_tab_list_order_reaches_the_report_when_unticked(tmp_path, monkeypatch):
    results = _record_results(monkeypatch, state_transitions, "save_state_transition_report")
    tab = _state_tab_with_data(tmp_path)
    assert tab.transition_node_order.order() == _SAVED_ORDER

    tab.chk_state_transition_auto_order.setChecked(False)
    tab.transition_node_order.set_order(_CUSTOM_ORDER)
    tab._on_state_transition()

    assert results[0]["circular_node_order"] == _CUSTOM_ORDER


def test_state_tab_offers_the_last_auto_arranged_order_for_fine_tuning(tmp_path, monkeypatch):
    results = _record_results(monkeypatch, state_transitions, "save_state_transition_report")
    tab = _state_tab_with_data(tmp_path)
    tab._on_state_transition()
    auto_order = results[0]["circular_node_order"]
    assert auto_order != _SAVED_ORDER
    assert tab.transition_node_order.order() == _SAVED_ORDER  # not overwritten behind the user's back

    tab.chk_state_transition_auto_order.setChecked(False)
    tab.transition_node_order.btn_use_auto.click()
    assert tab.transition_node_order.order() == auto_order


def test_track_tab_list_order_reaches_the_report_when_unticked(tmp_path, monkeypatch):
    _qt_app()
    from behav3d.napari._single_cell import TrackClassificationSubTab

    results = _record_results(monkeypatch, transition_analysis, "save_window_cluster_transition_analysis")
    loader = SimpleNamespace(output_dir=str(tmp_path), behav3d_parameters={}, metadata=None)
    tab = TrackClassificationSubTab(viewer=None, metadata_loader=loader, cell_type_getter=lambda: "tcells")
    _run_now(tab)
    tab._track_adata = _track_adata()
    tab._refresh_transition_node_order(tab._track_adata, "ClusterID")
    assert tab.transition_node_order.order() == _SAVED_ORDER

    tab.chk_transanalysis_include_window_sankey.setChecked(False)
    tab.chk_transanalysis_auto_order.setChecked(False)
    tab.transition_node_order.set_order(_CUSTOM_ORDER)
    tab._on_transition_analysis()

    assert results[0]["circular_node_order"] == _CUSTOM_ORDER


def _arrow_styles(ax, probs):
    """{probability: (linewidth, alpha)} of the drawn arrows, which are drawn weakest first."""
    drawn = sorted(p for p in probs.to_numpy().ravel() if np.isfinite(p) and p > 0)[-_drawn_arrow_count(ax):]
    arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    return {p: (arrow.get_linewidth(), arrow.get_alpha()) for p, arrow in zip(drawn, arrows)}


@pytest.mark.parametrize("scale_mode", ["relative", "absolute"])
@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_the_cutoff_removes_arrows_without_changing_the_others(scale_mode, gamma):
    probs = _random_probs(6, seed=9)
    colors = {label: "#000000" for label in probs.index}
    fig, (left, right) = plt.subplots(1, 2)
    for ax, cutoff in ((left, 0.0), (right, 0.2)):
        ctd._draw_circular_transition_diagram_on_ax(
            ax, probs, colors=colors, min_prob_to_draw=cutoff, emphasis_gamma=gamma, scale_mode=scale_mode,
        )
    all_arrows, kept_arrows = _arrow_styles(left, probs), _arrow_styles(right, probs)
    plt.close(fig)

    assert len(kept_arrows) == int((probs >= 0.2).sum().sum()) < len(all_arrows)
    for p, style in kept_arrows.items():
        assert style == pytest.approx(all_arrows[p]), p


def test_pair_figure_panels_show_shared_arrows_identically():
    probs = _random_probs(6, seed=10)
    fig = ctd.plot_circular_transition_diagram_pair(
        probs, state_order=list(probs.index), min_prob_to_draw=0.2, emphasis_gamma=1.0,
    )
    all_arrows, kept_arrows = (_arrow_styles(ax, probs) for ax in fig.axes[:2])
    plt.close(fig)
    assert 0 < len(kept_arrows) < len(all_arrows)
    for p, style in kept_arrows.items():
        assert style == pytest.approx(all_arrows[p]), p


def test_a_grid_panel_keeps_its_own_scale():
    # The strongest arrow of one cluster's panel is bold even if the matrix has stronger ones.
    labels = list("abc")
    probs = pd.DataFrame(0.0, index=labels, columns=labels)
    probs.loc["a", "b"] = 0.9
    probs.loc["b", "c"] = 0.3
    fig, ax = plt.subplots()
    ctd._draw_circular_transition_diagram_on_ax(
        ax, probs, colors={label: "#000000" for label in labels}, min_prob_to_draw=0.0, source_cluster="b",
    )
    arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    assert len(arrows) == 1 and arrows[0].get_linewidth() == pytest.approx(7.0)
    plt.close(fig)
