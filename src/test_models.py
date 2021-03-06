import models
import ciw
import numpy as np
import itertools


def test_simulation_builds_and_terminates():
    """
    Tests that the simulation is built properly and it terminates.
    """
    max_simulation_time = 100
    num_classes = 2
    num_servers = 2
    arrival_rates = [5, 5]
    service_rates = [3, 4]
    class_change_rate_matrix = [[None, 3], [2, None]]
    Q = models.build_and_run_simulation(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        class_change_rate_matrix=class_change_rate_matrix,
        max_simulation_time=max_simulation_time,
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
    assert (
        Q.network.customer_classes[0].class_change_time_distributions[0]
        == class_change_rate_matrix[0][0]
    )
    assert (
        Q.network.customer_classes[0].class_change_time_distributions[1].rate
        == class_change_rate_matrix[0][1]
    )
    assert (
        Q.network.customer_classes[1].class_change_time_distributions[0].rate
        == class_change_rate_matrix[1][0]
    )
    assert (
        Q.network.customer_classes[1].class_change_time_distributions[1]
        == class_change_rate_matrix[1][1]
    )
    assert len(inds) > 0


def test_write_state_space_for_states():
    """
    Tests that the state markov chain's state space is written correctly.
    """
    bound = 3
    n_classes = 2
    states = models.write_state_space_for_states(num_classes=n_classes, bound=bound)
    assert len(states) == bound**n_classes
    assert max(max(s) for s in states) == (bound - 1)
    assert min(min(s) for s in states) == 0
    assert all(len(s) == n_classes for s in states)

    bound = 10
    n_classes = 7
    states = models.write_state_space_for_states(num_classes=n_classes, bound=bound)
    assert len(states) == bound**n_classes
    assert max(max(s) for s in states) == (bound - 1)
    assert min(min(s) for s in states) == 0
    assert all(len(s) == n_classes for s in states)


def test_write_state_space_for_sojourn():
    """
    Tests that the sojourn markov chain's state space is written correctly.
    """
    bound = 3
    n_classes = 2
    states = models.write_state_space_for_sojourn(num_classes=n_classes, bound=bound)
    assert len(states) == (bound ** (n_classes + 1)) * n_classes + 1
    assert max(max(s) for s in states[:-1]) == (bound - 1)
    assert min(min(s) for s in states[:-1]) == 0
    assert all(len(s) == n_classes + 2 for s in states[:-1])
    assert states[-1] == "*"

    bound = 7
    n_classes = 3
    states = models.write_state_space_for_sojourn(num_classes=n_classes, bound=bound)
    assert len(states) == (bound ** (n_classes + 1)) * n_classes + 1
    assert max(max(s) for s in states[:-1]) == (bound - 1)
    assert min(min(s) for s in states[:-1]) == 0
    assert all(len(s) == n_classes + 2 for s in states[:-1])
    assert states[-1] == "*"


def test_get_all_pairs_of_non_zero_entries_states_example():
    """
    Tests we get all expected pairs of states that have a possible non-zero
    rate.
    """
    num_classes = 2
    bound = 3
    all_states = models.write_state_space_for_states(
        num_classes=num_classes, bound=bound
    )
    all_pairs_indices = models.get_all_pairs_of_non_zero_entries_states(
        State_Space=all_states, bound=bound
    )

    expected_pairs = [
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((0, 1), (0, 2)),
        ((0, 1), (0, 0)),
        ((0, 1), (1, 0)),
        ((0, 2), (1, 2)),
        ((0, 2), (0, 1)),
        ((0, 2), (1, 1)),
        ((1, 0), (2, 0)),
        ((1, 0), (0, 0)),
        ((1, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (2, 1)),
        ((1, 1), (0, 1)),
        ((1, 1), (0, 2)),
        ((1, 1), (1, 2)),
        ((1, 1), (1, 0)),
        ((1, 1), (2, 0)),
        ((1, 2), (2, 2)),
        ((1, 2), (0, 2)),
        ((1, 2), (1, 1)),
        ((1, 2), (2, 1)),
        ((2, 0), (1, 0)),
        ((2, 0), (1, 1)),
        ((2, 0), (2, 1)),
        ((2, 1), (1, 1)),
        ((2, 1), (1, 2)),
        ((2, 1), (2, 2)),
        ((2, 1), (2, 0)),
        ((2, 2), (1, 2)),
        ((2, 2), (2, 1)),
    ]
    expected_pairs_indices = [
        (all_states.index(s1), all_states.index(s2)) for s1, s2 in expected_pairs
    ]
    assert len(all_pairs_indices) == len(expected_pairs)
    assert set(expected_pairs_indices) == set(all_pairs_indices)


def test_get_all_pairs_of_non_zero_entries_sojourn_example():
    """
    Tests we get all expected pairs of states that have a possible non-zero
    rate.
    """
    num_classes = 2
    bound = 3
    all_states = models.write_state_space_for_sojourn(
        num_classes=num_classes, bound=bound
    )
    all_pairs_indices = models.get_all_pairs_of_non_zero_entries_sojourn(
        State_Space=all_states, bound=bound
    )
    assert len(all_pairs_indices) == 342


def test_states_transition_rates_everyone_in_service():
    """
    Tests the correct transition rates are given for state pairs, for state model, when everyone is in service
    """
    all_states = models.write_state_space_for_states(num_classes=2, bound=10)
    arrival_rates = [5, 7]
    service_rates = [3, 4]
    num_servers = 100
    thetas = [[None, 0.5], [0.2, None]]
    state = (10, 21)
    state_arrival1 = (11, 21)
    state_arrival2 = (10, 22)
    state_service1 = (9, 21)
    state_service2 = (10, 20)
    state_change12 = (9, 22)
    state_change21 = (11, 20)
    nonzero = [
        state,
        state_arrival1,
        state_arrival2,
        state_change12,
        state_change21,
        state_service1,
        state_service2,
    ]
    assert (
        models.find_transition_rates_for_states(
            state, state_arrival1, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[0]
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_arrival2, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[1]
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_service1, num_servers, arrival_rates, service_rates, thetas
        )
        == state[0] * service_rates[0]
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_service2, num_servers, arrival_rates, service_rates, thetas
        )
        == state[1] * service_rates[1]
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_change12, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_change21, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    for next_state in all_states:
        if next_state not in nonzero:
            assert (
                models.find_transition_rates_for_states(
                    state, next_state, num_servers, arrival_rates, service_rates, thetas
                )
                == 0
            )


def test_states_transition_rates_noone_in_service():
    """
    Tests the correct transition rates are given for state pairs, for state model, when no-one is in service
    """
    all_states = models.write_state_space_for_states(num_classes=2, bound=10)
    arrival_rates = [5, 7]
    service_rates = [3, 4]
    num_servers = 0
    thetas = [[None, 0.5], [0.2, None]]
    state = (10, 21)
    state_arrival1 = (11, 21)
    state_arrival2 = (10, 22)
    state_service1 = (9, 21)
    state_service2 = (10, 20)
    state_change12 = (9, 22)
    state_change21 = (11, 20)
    nonzero = [
        state,
        state_arrival1,
        state_arrival2,
        state_change12,
        state_change21,
        state_service1,
        state_service2,
    ]
    assert (
        models.find_transition_rates_for_states(
            state, state_arrival1, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[0]
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_arrival2, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[1]
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_service1, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_service2, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_change12, num_servers, arrival_rates, service_rates, thetas
        )
        == state[0] * thetas[0][1]
    )
    assert (
        models.find_transition_rates_for_states(
            state, state_change21, num_servers, arrival_rates, service_rates, thetas
        )
        == state[1] * thetas[1][0]
    )
    for next_state in all_states:
        if next_state not in nonzero:
            assert (
                models.find_transition_rates_for_states(
                    state, next_state, num_servers, arrival_rates, service_rates, thetas
                )
                == 0
            )


def test_sojourn_transition_rates_everyone_in_service():
    """
    Tests the correct transition rates are given for state pairs, for sojourn model, when everyone is in service
    """
    all_states = models.write_state_space_for_sojourn(num_classes=3, bound=22)
    arrival_rates = [5, 7, 3]
    service_rates = [3, 4, 9]
    num_servers = 100
    thetas = [[None, 0.5, 0.3], [0.2, None, 0.1], [0.4, 0.2, None]]
    state = (11, 21, 7, 14, 1)
    state_arrival1 = (12, 21, 7, 14, 1)
    state_arrival2 = (11, 21, 7, 15, 1)
    state_arrival3 = (11, 21, 8, 14, 1)
    state_service1 = (10, 21, 7, 14, 1)
    state_service2front = (11, 20, 7, 14, 1)
    state_service2behind = (11, 21, 7, 13, 1)
    state_service3 = (11, 21, 6, 14, 1)
    state_service_me = "*"
    state_change12 = (10, 21, 7, 15, 1)
    state_change13 = (10, 21, 8, 14, 1)
    state_change21front = (12, 20, 7, 14, 1)
    state_change23front = (11, 20, 8, 14, 1)
    state_change21behind = (12, 21, 7, 13, 1)
    state_change23behind = (11, 21, 8, 13, 1)
    state_change31 = (12, 21, 6, 14, 1)
    state_change32 = (11, 21, 6, 15, 1)
    state_change_me1 = (11, 35, 7, 0, 0)
    state_change_me3 = (11, 35, 7, 0, 2)
    nonzero = [
        state,
        state_arrival1,
        state_arrival2,
        state_arrival3,
        state_change12,
        state_change13,
        state_change21behind,
        state_change21front,
        state_change23behind,
        state_change23front,
        state_change31,
        state_change32,
        state_change_me1,
        state_change_me3,
        state_service1,
        state_service2behind,
        state_service2front,
        state_service3,
        state_service_me,
    ]
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_arrival1, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[0]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_arrival2, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[1]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_arrival3, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[2]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_service1, num_servers, arrival_rates, service_rates, thetas
        )
        == state[0] * service_rates[0]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_service2front,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == state[1] * service_rates[1]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_service2behind,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == state[-2] * service_rates[1]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_service3, num_servers, arrival_rates, service_rates, thetas
        )
        == state[2] * service_rates[2]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_service_me, num_servers, arrival_rates, service_rates, thetas
        )
        == service_rates[1]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change12, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change13, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change21front,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change23front,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change21behind,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change23behind,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change31, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change32, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change_me1, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change_me3, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    for next_state in all_states:
        if next_state not in nonzero:
            assert (
                models.find_transition_rates_for_sojourn_time(
                    state, next_state, num_servers, arrival_rates, service_rates, thetas
                )
                == 0
            )


def test_sojourn_transition_rates_noone_in_service():
    """
    Tests the correct transition rates are given for state pairs, for sojourn model, when no-one is in service
    """
    all_states = models.write_state_space_for_sojourn(num_classes=3, bound=22)
    arrival_rates = [5, 7, 3]
    service_rates = [3, 4, 9]
    num_servers = 0
    thetas = [[None, 0.5, 0.3], [0.2, None, 0.1], [0.4, 0.2, None]]
    state = (11, 21, 7, 14, 1)
    state_arrival1 = (12, 21, 7, 14, 1)
    state_arrival2 = (11, 21, 7, 15, 1)
    state_arrival3 = (11, 21, 8, 14, 1)
    state_service1 = (10, 21, 7, 14, 1)
    state_service2front = (11, 20, 7, 14, 1)
    state_service2behind = (11, 21, 7, 13, 1)
    state_service3 = (11, 21, 6, 14, 1)
    state_service_me = "*"
    state_change12 = (10, 21, 7, 15, 1)
    state_change13 = (10, 21, 8, 14, 1)
    state_change21front = (12, 20, 7, 14, 1)
    state_change23front = (11, 20, 8, 14, 1)
    state_change21behind = (12, 21, 7, 13, 1)
    state_change23behind = (11, 21, 8, 13, 1)
    state_change31 = (12, 21, 6, 14, 1)
    state_change32 = (11, 21, 6, 15, 1)
    state_change_me1 = (11, 35, 7, 0, 0)
    state_change_me3 = (11, 35, 7, 0, 2)
    nonzero = [
        state,
        state_arrival1,
        state_arrival2,
        state_arrival3,
        state_change12,
        state_change13,
        state_change21behind,
        state_change21front,
        state_change23behind,
        state_change23front,
        state_change31,
        state_change32,
        state_change_me1,
        state_change_me3,
        state_service1,
        state_service2behind,
        state_service2front,
        state_service3,
        state_service_me,
    ]
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_arrival1, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[0]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_arrival2, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[1]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_arrival3, num_servers, arrival_rates, service_rates, thetas
        )
        == arrival_rates[2]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_service1, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_service2front,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_service2behind,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_service3, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_service_me, num_servers, arrival_rates, service_rates, thetas
        )
        == 0
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change12, num_servers, arrival_rates, service_rates, thetas
        )
        == state[0] * thetas[0][1]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change13, num_servers, arrival_rates, service_rates, thetas
        )
        == state[0] * thetas[0][2]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change21front,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == state[1] * thetas[1][0]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change23front,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == state[1] * thetas[1][2]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change21behind,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == state[-2] * thetas[1][0]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state,
            state_change23behind,
            num_servers,
            arrival_rates,
            service_rates,
            thetas,
        )
        == state[-2] * thetas[1][2]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change31, num_servers, arrival_rates, service_rates, thetas
        )
        == state[2] * thetas[2][0]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change32, num_servers, arrival_rates, service_rates, thetas
        )
        == state[2] * thetas[2][1]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change_me1, num_servers, arrival_rates, service_rates, thetas
        )
        == thetas[1][0]
    )
    assert (
        models.find_transition_rates_for_sojourn_time(
            state, state_change_me3, num_servers, arrival_rates, service_rates, thetas
        )
        == thetas[1][2]
    )
    for next_state in all_states:
        if next_state not in nonzero:
            assert (
                models.find_transition_rates_for_sojourn_time(
                    state, next_state, num_servers, arrival_rates, service_rates, thetas
                )
                == 0
            )


def test_writes_transition_matrix_states():
    """
    Tests that a valid transition matrix is written.
    """
    bound = 5
    num_classes = 2
    State_Space = models.write_state_space_for_states(
        num_classes=num_classes, bound=bound
    )
    transition_matrix = models.write_transition_matrix(
        State_Space=State_Space,
        transition_function=models.find_transition_rates_for_states,
        non_zero_pair_function=models.get_all_pairs_of_non_zero_entries_states,
        num_servers=3,
        arrival_rates=[14, 10],
        service_rates=[7, 5],
        thetas=[[None, 5], [3, None]],
        bound=bound,
    )
    matrix_size = bound**num_classes
    assert transition_matrix.shape == (matrix_size, matrix_size)
    assert transition_matrix.size == matrix_size**2
    assert (transition_matrix.sum(axis=1) == [0.0 for _ in range(matrix_size)]).all()
    diag = transition_matrix.diagonal().copy()
    np.fill_diagonal(transition_matrix, np.zeros(25))
    assert all(diag < 0)
    assert all(transition_matrix.sum(axis=1) == -diag)


def test_writes_transition_matrix_sojourn():
    """
    Tests that a valid absorbing transition matrix is written.
    """
    bound = 4
    num_classes = 2
    State_Space = models.write_state_space_for_sojourn(
        num_classes=num_classes, bound=bound
    )
    transition_matrix = models.write_transition_matrix(
        State_Space=State_Space,
        transition_function=models.find_transition_rates_for_sojourn_time,
        non_zero_pair_function=models.get_all_pairs_of_non_zero_entries_sojourn,
        num_servers=3,
        arrival_rates=[14, 10],
        service_rates=[7, 5],
        thetas=[[None, 5], [3, None]],
        bound=bound,
    )
    matrix_size = ((bound ** (num_classes + 1)) * num_classes) + 1
    assert transition_matrix.shape == (matrix_size, matrix_size)
    assert transition_matrix.size == matrix_size**2
    assert (transition_matrix.sum(axis=1) == [0.0 for _ in range(matrix_size)]).all()
    assert (transition_matrix[-1] == [0.0 for _ in range(matrix_size)]).all()
    diag = transition_matrix.diagonal().copy()
    np.fill_diagonal(transition_matrix, np.zeros(25))
    assert all(diag <= 0)
    assert all(transition_matrix.sum(axis=1) == -diag)


def test_get_numbers_in_service():
    """
    Tests we correctly give the number of customers of that class in service.
    """
    state = (0, 3, 2, 1, 0)
    # 0 servers
    in_service = [models.get_numbers_in_service(state, clss, 0, 3) for clss in range(4)]
    assert in_service == [0, 0, 0, 0]
    # 1 servers
    in_service = [models.get_numbers_in_service(state, clss, 1, 3) for clss in range(4)]
    assert in_service == [0, 0, 0, 0]
    # 2 servers
    in_service = [models.get_numbers_in_service(state, clss, 2, 3) for clss in range(4)]
    assert in_service == [0, 0, 0, 1]
    # 3 servers
    in_service = [models.get_numbers_in_service(state, clss, 3, 3) for clss in range(4)]
    assert in_service == [0, 1, 0, 1]
    # 4 servers
    in_service = [models.get_numbers_in_service(state, clss, 4, 3) for clss in range(4)]
    assert in_service == [0, 2, 0, 1]
    # 5 servers
    in_service = [models.get_numbers_in_service(state, clss, 5, 3) for clss in range(4)]
    assert in_service == [0, 3, 0, 1]
    # 6 servers
    in_service = [models.get_numbers_in_service(state, clss, 6, 3) for clss in range(4)]
    assert in_service == [0, 3, 1, 1]
    # 7 servers
    in_service = [models.get_numbers_in_service(state, clss, 7, 3) for clss in range(4)]
    assert in_service == [0, 3, 2, 1]
    # 8 servers
    in_service = [models.get_numbers_in_service(state, clss, 8, 3) for clss in range(4)]
    assert in_service == [0, 3, 2, 1]

    state = (2, 1, 2, 2, 1)
    # 0 servers
    in_service = [models.get_numbers_in_service(state, clss, 0, 3) for clss in range(4)]
    assert in_service == [0, 0, 0, 0]
    # 1 servers
    in_service = [models.get_numbers_in_service(state, clss, 1, 3) for clss in range(4)]
    assert in_service == [1, 0, 0, 0]
    # 2 servers
    in_service = [models.get_numbers_in_service(state, clss, 2, 3) for clss in range(4)]
    assert in_service == [2, 0, 0, 0]
    # 3 servers
    in_service = [models.get_numbers_in_service(state, clss, 3, 3) for clss in range(4)]
    assert in_service == [2, 1, 0, 0]
    # 4 servers
    in_service = [models.get_numbers_in_service(state, clss, 4, 3) for clss in range(4)]
    assert in_service == [2, 1, 0, 0]
    # 5 servers
    in_service = [models.get_numbers_in_service(state, clss, 5, 3) for clss in range(4)]
    assert in_service == [2, 1, 0, 1]
    # 6 servers
    in_service = [models.get_numbers_in_service(state, clss, 6, 3) for clss in range(4)]
    assert in_service == [2, 1, 0, 2]
    # 7 servers
    in_service = [models.get_numbers_in_service(state, clss, 7, 3) for clss in range(4)]
    assert in_service == [2, 1, 1, 2]
    # 8 servers
    in_service = [models.get_numbers_in_service(state, clss, 8, 3) for clss in range(4)]
    assert in_service == [2, 1, 2, 2]


def test_correct_rates_in_transition_matrices():
    """
    Tests that the correct rates are places in the correct places in the trandsition matrices
    """
    num_classes = 2
    num_servers = 2
    arrival_rates = [5, 5]
    service_rates = [3, 4]
    class_change_rate_matrix = [[None, 3], [2, None]]
    bound = 12
    State_Space = models.write_state_space_for_states(
        num_classes=num_classes, bound=bound
    )
    transition_matrix = models.write_transition_matrix(
        State_Space=State_Space,
        transition_function=models.find_transition_rates_for_states,
        non_zero_pair_function=models.get_all_pairs_of_non_zero_entries_states,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=class_change_rate_matrix,
        bound=bound,
    )
    nonzero_pairs = models.get_all_pairs_of_non_zero_entries_states(State_Space, bound)
    size_mat = len(State_Space)
    for i, j in itertools.product(range(size_mat), range(size_mat)):
        if ((i, j) not in nonzero_pairs) and (i != j):
            assert transition_matrix[i][j] == 0
