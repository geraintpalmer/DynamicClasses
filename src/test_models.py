import models
import ciw


def test_simulation_builds_and_terminates():
    """
    TODO Document and add to this test.
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
