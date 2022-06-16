import models
import ciw


def test_simulation_builds_and_terminates():
    """
    Tests that the simulation is built properly and it terminates.
    """
    max_simulation_time = 100
    num_classes = 2
    num_servers = 2
    arrival_rates=[5, 5]
    service_rates=[3, 4]
    class_change_rate_matrix=[[None, 3], [2, None]]
    Q = models.build_and_run_simulation(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        class_change_rate_matrix=class_change_rate_matrix,
        max_simulation_time=max_simulation_time
    )
    inds = Q.get_all_individuals()
    assert Q.current_time > max_simulation_time
    assert isinstance(Q, ciw.Simulation)
    assert len(Q.transitive_nodes) == 1
    assert Q.nodes[1].c == num_servers
    assert Q.service_times[1][0].rate == service_rates[0]
    assert Q.service_times[1][1].rate == service_rates[1]
    assert Q.inter_arrival_times[1][0].rate == arrival_rates[0]
    assert Q.inter_arrival_times[1][1].rate == arrival_rates[1]
    assert Q.network.customer_classes[0].class_change_time_distributions[0] == class_change_rate_matrix[0][0]
    assert Q.network.customer_classes[0].class_change_time_distributions[1].rate == class_change_rate_matrix[0][1]
    assert Q.network.customer_classes[1].class_change_time_distributions[0].rate == class_change_rate_matrix[1][0]
    assert Q.network.customer_classes[1].class_change_time_distributions[1] == class_change_rate_matrix[1][1]
    assert len(inds) > 0


def test_write_state_space_for_states():
    """
    Tests that the state markov chain's state space is written correctly.
    """
    infty = 3
    n_classes = 2
    states = models.write_state_space_for_states(num_classes=n_classes, infty=infty)
    assert len(states) == infty ** n_classes
    assert max(max(s) for s in states) == (infty - 1)
    assert min(min(s) for s in states) == 0
    assert all(len(s) == n_classes for s in states)

    infty = 10
    n_classes = 7
    states = models.write_state_space_for_states(num_classes=n_classes, infty=infty)
    assert len(states) == infty ** n_classes
    assert max(max(s) for s in states) == (infty - 1)
    assert min(min(s) for s in states) == 0
    assert all(len(s) == n_classes for s in states)


def test_write_state_space_for_sojourn():
    """
    Tests that the sojourn markov chain's state space is written correctly.
    """
    infty = 3
    n_classes = 2
    states = models.write_state_space_for_sojourn(num_classes=n_classes, infty=infty)
    assert len(states) == (infty ** (n_classes + 1)) * n_classes + 1
    assert max(max(s) for s in states[:-1]) == (infty - 1)
    assert min(min(s) for s in states[:-1]) == 0
    assert all(len(s) == n_classes + 2 for s in states[:-1])
    assert states[-1] == '*'

    infty = 7
    n_classes = 3
    states = models.write_state_space_for_sojourn(num_classes=n_classes, infty=infty)
    assert len(states) == (infty ** (n_classes + 1)) * n_classes + 1
    assert max(max(s) for s in states[:-1]) == (infty - 1)
    assert min(min(s) for s in states[:-1]) == 0
    assert all(len(s) == n_classes + 2 for s in states[:-1])
    assert states[-1] == '*'
