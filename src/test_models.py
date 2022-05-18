import models


def test_simulation_terminates():
    """
    TODO Document and add to this test.
    """
    S = models.Simulation(
        number_of_servers=1,
        arrival_rates=[5, 6],
        service_rates=[12, 13],
        class_change_rate_matrix=[[None, 3], [1, None]],
        preempt="resample",
        max_simulation_time=200,
        warmup=20,
    )
    assert round(S.probs[((0, 0),)], 2) == 0.12
