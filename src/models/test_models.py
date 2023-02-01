import models
import ciw
import numpy as np
import itertools


def test_get_state_probabilites():
    """
    Tests that the state probabilities are calculated correctly.
    """
    num_classes = 2
    num_servers = 2
    arrival_rates = [5, 7]
    service_rates = [6, 4]
    thetas = [
        [None, 2],
        [4, None],
    ]
    bound = 3
    calculated_state_probs = models.get_state_probabilities(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )

    expected_state_probs = {
        (0, 0): 0.09087913672922397,
        (0, 1): 0.1735865446485757,
        (0, 2): 0.16908761375045728,
        (1, 0): 0.06603391035939747,
        (1, 1): 0.13142164121149752,
        (1, 2): 0.16383886103598583,
        (2, 0): 0.017377344831420383,
        (2, 1): 0.07547921389586176,
        (2, 2): 0.11229573353758014,
    }

    for (key_1, value_1), (key_2, value_2) in zip(
        calculated_state_probs.items(), expected_state_probs.items()
    ):
        assert key_1 == key_2
        assert np.round(value_1, 5) == np.round(value_2, 5)


def test_build_state_space_and_transition_matrix_sojourn_mc():
    """
    Tests that the state space and transition matrix are built correctly.
    """
    num_classes = 2
    num_servers = 2
    arrival_rates = [5, 7]
    service_rates = [6, 4]
    thetas = [
        [None, 2],
        [4, None],
    ]
    bound = 1

    (
        calculated_state_space,
        calculated_transition_matrix,
    ) = models.build_state_space_and_transition_matrix_sojourn_mc(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )

    expected_state_space = [(0, 0, 0, 0), (0, 0, 0, 1), "*"]
    for calcualted_state, expected_state in zip(
        calculated_state_space, expected_state_space
    ):
        assert calcualted_state == expected_state

    expected_transition_matrix = np.array(
        [[-6.0, 0.0, 6.0], [0.0, -4.0, 4.0], [0.0, 0.0, 0.0]]
    )
    assert np.allclose(calculated_transition_matrix, expected_transition_matrix)


def test_get_mean_sojourn_times():
    """
    Tests that the mean sojourn times are calculated correctly.
    """
    num_classes = 2
    num_servers = 2
    arrival_rates = [5, 7]
    service_rates = [6, 4]
    thetas = [
        [None, 2],
        [4, None],
    ]
    bound = 2

    state_probs = models.get_state_probabilities(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )
    (
        state_space,
        transition_matrix,
    ) = models.build_state_space_and_transition_matrix_sojourn_mc(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )
    calculated_sojourn_times = models.get_mean_sojourn_times(
        state_space, transition_matrix, num_classes, arrival_rates, state_probs
    )
    expected_sojourn_times = [
        0.16666666666666666,
        0.29127484397130005,
        0.23935477009436945,
    ]

    for time_1, time_2 in zip(calculated_sojourn_times, expected_sojourn_times):
        assert np.round(time_1, 5) == np.round(time_2, 5)


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


def test_find_hitting_probs():
    """
    Made up example based on the maths of https://youtu.be/edTup9lQU90, but now treating
    transient states as absorbing states:

    P = (
      (1/5,  1/5,  1/5,  2/5,  0  ,  0  )
      (0  ,  1  ,  0  ,    0,  0  ,  0  )
      (0  ,  1/3,  0  ,  1/3,  1/3,  0  )
      (0  ,  0  ,  0  ,    1,  0  ,  0  )
      (1/2,  0  ,  0  ,    0,  0  ,  1/2)
      (0  ,  0  ,  1/2,  1/4,  1/4,  0  )
    )
    + There are two abosrbing states, state 1 and 3, so h_{10} = h_{30} = 0 by definition.
    + For $h_{00}$ we are already at state 0, so guaranteed hit, h_{00} = 1
    + To find h_{20}, h_{40}, and h_{50} we solve:

      h_{20} &= (1/3)h_{10} + (1/3)h_{30} + (1/3)h_{40}
      h_{40} &= (1/2)h_{00} + (1/2)h_{50}
      h_{50} &= (1/2)h_{20} + (1/4)h_{30} + (1/4)h_{40}

      which simplifies to:

      h_{20} &= (1/3)h_{40}
      h_{40} &= (1/2) +(1/2)h_{50}
      h_{50} &= (1/2)h_{40} +(1/4)h_{40}

    + This gives
      - h_{20} = 4/19 = 0.210526
      - h_{40} = 12/19 = 0.631579
      - h_{50} = 5/19 = 0.263158
    """
    P = np.array(
        [
            [1 / 5, 1 / 5, 1 / 5, 2 / 5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1 / 3, 0.0, 1 / 3, 1 / 3, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1 / 2, 0.0, 0.0, 0.0, 0.0, 1 / 2],
            [0.0, 0.0, 1 / 2, 1 / 4, 1 / 4, 0.0],
        ]
    )
    probs = models.find_hitting_probs(range(6), P, [0])
    assert round(probs[0], 6) == 1
    assert round(probs[2], 6) == 0.210526
    assert round(probs[4], 6) == 0.631579
    assert round(probs[5], 6) == 0.263158


def build_state_space_and_transition_matrix(boundary):
    classes = 2
    num_of_servers = 3
    arrival_rates = [1, 2]
    service_rates = [100, 100]
    thetas = [[None, 1], [10, None]]
    state_space = models.write_state_space_for_sojourn(
        num_classes=classes, bound=boundary
    )
    transition_matrix = models.write_transition_matrix(
        State_Space=state_space,
        transition_function=models.find_transition_rates_for_sojourn_time,
        non_zero_pair_function=models.get_all_pairs_of_non_zero_entries_sojourn,
        num_servers=num_of_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=boundary,
    )
    return state_space, transition_matrix


def test_bound_check_bound_increase():
    expected = [False, False, False, False, False, True, True, True, True, True]
    for i, boundary in enumerate(range(5, 15)):
        state_space, transition_matrix = build_state_space_and_transition_matrix(
            boundary=boundary
        )
        condition = models.bound_check(
            state_space, transition_matrix, boundary, 0.63, 0.01
        )
        assert condition == expected[i], (boundary, i)


def test_bound_check_reasonable_ratio_increase():
    expected = [True, True, True, False, False, False, False, False, False, False]
    for i, ratio in enumerate(np.arange(0.6, 0.7, 0.01)):
        state_space, transition_matrix = build_state_space_and_transition_matrix(
            boundary=9
        )
        condition = models.bound_check(state_space, transition_matrix, 9, ratio, 0.01)
        assert condition == expected[i]


def test_bound_check_epsilon_decrease():
    expected = [False, False, False, False, True, True, True, True, True, True]
    for i, epsilon in enumerate(np.arange(0.01, 0.1, 0.01)):
        state_space, transition_matrix = build_state_space_and_transition_matrix(
            boundary=9
        )
        condition = models.bound_check(state_space, transition_matrix, 9, 0.7, epsilon)
        assert condition == expected[i]


def test_num_of_customers_from_state_probs():
    """
    Test the function that calculates the average number of customers in the system
    from the state probabilities.
    """
    state_probs = {
        (0, 0, 0): 0.5000000000000001,
        (0, 0, 1): 0.25000000000000006,
        (0, 1, 0): 0.08333333333333336,
        (0, 1, 1): 0.04166666666666668,
        (1, 0, 0): 0.07142857142857144,
        (1, 0, 1): 0.03571428571428571,
        (1, 1, 0): 0.01190476190476191,
        (1, 1, 1): 0.005952380952380958,
    }

    expected_mean_num_customers = [
        0.125,
        0.1428571428571429,
        0.33333333333333337,
        0.6011904761904764,
    ]
    expected_variance_num_customers = [
        0.10937500000000003,
        0.12244897959183679,
        0.2222222222222223,
        0.4540462018140592,
    ]
    mean_num_customers = models.get_average_num_of_customers_from_state_probs(
        state_probs=state_probs, num_classes=3
    )
    variance_num_customers = (
        models.get_variance_of_number_of_customers_from_state_probs(
            state_probs=state_probs, average_in_system=mean_num_customers, num_classes=3
        )
    )

    for calculated, expected in zip(mean_num_customers, expected_mean_num_customers):
        assert round(calculated, 6) == round(expected, 6)

    for calculated, expected in zip(
        variance_num_customers, expected_variance_num_customers
    ):
        assert round(calculated, 6) == round(expected, 6)


def test_num_of_waiting_customers_from_state_probs():
    """
    Test the function that calculates the variance of the number of customers in
    the system from the state probabilities.
    """
    state_probs = {
        (0, 0, 0): 0.43068862456851414,
        (0, 0, 1): 0.17290069162270447,
        (0, 0, 2): 0.02297705234495762,
        (0, 1, 0): 0.09050701021124082,
        (0, 1, 1): 0.032735754808111535,
        (0, 1, 2): 0.0043101664726587705,
        (0, 2, 0): 0.02139241253255558,
        (0, 2, 1): 0.007255033096773404,
        (0, 2, 2): 0.0011204853428618273,
        (1, 0, 0): 0.057603204170384066,
        (1, 0, 1): 0.02734945714377484,
        (1, 0, 2): 0.00442983328441701,
        (1, 1, 0): 0.03650285601961712,
        (1, 1, 1): 0.01339640744852269,
        (1, 1, 2): 0.0023589395923088413,
        (1, 2, 0): 0.009713525107870219,
        (1, 2, 1): 0.0041194757535687595,
        (1, 2, 2): 0.0009089379880510089,
        (2, 0, 0): 0.020763345305046668,
        (2, 0, 1): 0.00858953069181794,
        (2, 0, 2): 0.0017320743788380231,
        (2, 1, 0): 0.01396193587637746,
        (2, 1, 1): 0.005884932929408871,
        (2, 1, 2): 0.001339399319490708,
        (2, 2, 0): 0.004474057252765048,
        (2, 2, 1): 0.0023307074817494147,
        (2, 2, 2): 0.0006541492556130306,
    }

    mean_waiting = models.get_average_num_of_customers_waiting_from_state_probs(
        state_probs=state_probs, num_servers=1, num_classes=3
    )
    variance_waiting = models.get_variance_of_customers_waiting_from_state_probs(
        state_probs=state_probs,
        num_servers=1,
        average_waiting=mean_waiting,
        num_classes=3,
    )

    expecetd_mean_waiting = [
        0.05973013249110716,
        0.14761410783715148,
        0.1583463229671635,
        0.36569056329542216,
    ]
    expected_variance_waiting = [
        0.05616244376370194,
        0.17022588868382826,
        0.1669807362384207,
        0.5414451532512363,
    ]

    for calculated, expected in zip(mean_waiting, expecetd_mean_waiting):
        assert round(calculated, 6) == round(expected, 6)

    for calculated, expected in zip(variance_waiting, expected_variance_waiting):
        assert round(calculated, 6) == round(expected, 6)


def test_get_empty_probabilities_from_state_probs():
    """
    Test the function that calculates the probability of the system being empty
    from the state probabilities.
    """
    state_probs = {
        (0, 0, 0): 0.43068862456851414,
        (0, 0, 1): 0.17290069162270447,
        (0, 0, 2): 0.02297705234495762,
        (0, 1, 0): 0.09050701021124082,
        (0, 1, 1): 0.032735754808111535,
        (0, 1, 2): 0.0043101664726587705,
        (0, 2, 0): 0.02139241253255558,
        (0, 2, 1): 0.007255033096773404,
        (0, 2, 2): 0.0011204853428618273,
        (1, 0, 0): 0.057603204170384066,
        (1, 0, 1): 0.02734945714377484,
        (1, 0, 2): 0.00442983328441701,
        (1, 1, 0): 0.03650285601961712,
        (1, 1, 1): 0.01339640744852269,
        (1, 1, 2): 0.0023589395923088413,
        (1, 2, 0): 0.009713525107870219,
        (1, 2, 1): 0.0041194757535687595,
        (1, 2, 2): 0.0009089379880510089,
        (2, 0, 0): 0.020763345305046668,
        (2, 0, 1): 0.00858953069181794,
        (2, 0, 2): 0.0017320743788380231,
        (2, 1, 0): 0.01396193587637746,
        (2, 1, 1): 0.005884932929408871,
        (2, 1, 2): 0.001339399319490708,
        (2, 2, 0): 0.004474057252765048,
        (2, 2, 1): 0.0023307074817494147,
        (2, 2, 2): 0.0006541492556130306,
    }

    empty_probs = models.get_empty_probabilities_from_state_probs(
        state_probs=state_probs, num_classes=3
    )

    expected_empty_probs = [
        0.7838872310003782,
        0.7470338135104547,
        0.6856069710443712,
        0.43068862456851414,
    ]

    for calculated, expected in zip(empty_probs, expected_empty_probs):
        assert round(calculated, 6) == round(expected, 6)


def test_get_mean_sojourn_times_using_simulation():
    """
    Test the function that calculates the mean sojourn times using simulation.
    """
    num_classes = 3
    num_servers = 4
    arrival_rates = [1, 1, 1]
    service_rates = [7, 6, 2]
    thetas = [[None, 1, 2], [3, None, 1], [3, 3, None]]
    max_simulation_time = 10000
    warmup_time = max_simulation_time * 0.4
    cooldown_time = 100

    Q = models.build_and_run_simulation(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        class_change_rate_matrix=thetas,
        max_simulation_time=max_simulation_time,
    )

    sojourn_times = models.get_mean_sojourn_times_using_simulation(
        Q, max_simulation_time, warmup_time, cooldown_time, num_classes
    )

    expected_sojourn_times = [
        0.14016834542458317,
        0.16976994909878249,
        0.4935770593297915,
        0.26699596150196186,
    ]

    for calculated, expected in zip(sojourn_times, expected_sojourn_times):
        assert round(calculated, 6) == round(expected, 6)

def test_write_row_markov():
    """
    Tests the write row function ising the Markov chain.
    Tests if the bound iterates correctly.

    We know that the following parameters:
        num_classes=2
        arrival_rates=[5, 6]
        service_rates=[8, 10]
        num_servers=2
        thetas=[[None, 1], [1, None]]
    and the following hyperparameters:
        reasonable_ratio=0.25
        epsilon=0.01
    needs a bound of 9.
    """
    # Tests it reaches and stops at 9 when using steps of 1
    row = models.write_row_markov(
        num_classes=2,
        arrival_rates=[5, 6],
        service_rates=[8, 10],
        num_servers=2,
        thetas=[[None, 1], [1, None]],
        bound_initial=5,
        bound_final=14,
        bound_step=1,
        reasonable_ratio=0.25,
        epsilon=0.01,
    )
    assert row[0] == 9
    assert row[-1] < 0.01
    assert all(r is not None for r in row[11:-2])

    # Tests it reaches 9 but stops at 11 when using steps of 3
    row = models.write_row_markov(
        num_classes=2,
        arrival_rates=[5, 6],
        service_rates=[8, 10],
        num_servers=2,
        thetas=[[None, 1], [1, None]],
        bound_initial=5,
        bound_final=14,
        bound_step=3,
        reasonable_ratio=0.25,
        epsilon=0.01,
    )
    assert row[0] == 11
    assert row[-1] < 0.01
    assert all(r is not None for r in row[11:-2])

    # Tests it reaches and stops at 12 when starting at 11
    row = models.write_row_markov(
        num_classes=2,
        arrival_rates=[5, 6],
        service_rates=[8, 10],
        num_servers=2,
        thetas=[[None, 1], [1, None]],
        bound_initial=12,
        bound_final=17,
        bound_step=1,
        reasonable_ratio=0.25,
        epsilon=0.01,
    )
    assert row[0] == 12
    assert row[-1] < 0.01
    assert all(r is not None for r in row[11:-2])

    # Tests it doesn't reach 9 maximum is 7
    row = models.write_row_markov(
        num_classes=2,
        arrival_rates=[5, 6],
        service_rates=[8, 10],
        num_servers=2,
        thetas=[[None, 1], [1, None]],
        bound_initial=5,
        bound_final=7,
        bound_step=1,
        reasonable_ratio=0.25,
        epsilon=0.01,
    )
    assert row[0] == 7
    assert row[-1] > 0.01
    assert all(r is None for r in row[11:-2])
