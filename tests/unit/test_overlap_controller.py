from aitrainer.overlap import OverlapController


def test_overlap_controller_records_synchronous_baseline():
    controller = OverlapController()
    assert controller.submit("baseline", lambda: 7) == 7
    summary = controller.summary()
    assert summary["operations"] == 1.0
    assert summary["overlap_ratio"] == 0.0
