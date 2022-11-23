import numpy as np
import itertools
import ciw
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")


def get_state_probabilities(
    num_classes, num_servers, arrival_rates, service_rates, thetas, bound
):
    """
    Get's the system's state probabilities
    """
    State_Space = write_state_space_for_states(num_classes=num_classes, bound=bound)
    transition_matrix = write_transition_matrix(
        State_Space=State_Space,
        transition_function=find_transition_rates_for_states,
        non_zero_pair_function=get_all_pairs_of_non_zero_entries_states,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )
    probs = solve_probabilities(
        State_Space=State_Space, transition_matrix=transition_matrix
    )
    return probs


def build_state_space_and_transition_matrix_sojourn_mc(
    num_classes, num_servers, arrival_rates, service_rates, thetas, bound
):
    """
    Build the state space and transition matrix for the sojourn time Markov Chain.
    """
    State_Space = write_state_space_for_sojourn(num_classes=num_classes, bound=bound)
    transition_matrix = write_transition_matrix(
        State_Space=State_Space,
        transition_function=find_transition_rates_for_sojourn_time,
        non_zero_pair_function=get_all_pairs_of_non_zero_entries_sojourn,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )
    return State_Space, transition_matrix


def write_state_space_for_states(num_classes, bound):
    """
    Write the states for the state probability model
    """
    State_Space = list(itertools.product(*[range(bound) for _ in range(num_classes)]))
    return State_Space


def write_state_space_for_sojourn(num_classes, bound):
    """
    Write the states for the sojourn time model
    """
    State_Space = list(
        itertools.product(
            *(
                [range(bound) for _ in range(num_classes)]
                + [range(bound)]
                + [range(num_classes)]
            )
        )
    ) + ["*"]
    return State_Space


def find_transition_rates_for_states(
    state1, state2, num_servers, arrival_rates, service_rates, thetas
):
    """
    Finds the transition rates for given state transition
    """
    num_classes = len(arrival_rates)
    delta = tuple(state2[i] - state1[i] for i in range(num_classes))
    if delta.count(0) == num_classes - 1:
        if delta.count(-1) == 1:
            leaving_clss = delta.index(-1)
            if sum([state1[clss] for clss in range(leaving_clss)]) < num_servers:
                number_in_service = min(
                    num_servers - min(num_servers, sum(state1[:leaving_clss])),
                    state1[leaving_clss],
                )
                return service_rates[leaving_clss] * number_in_service
        elif delta.count(1) == 1:
            arriving_clss = delta.index(1)
            return arrival_rates[arriving_clss]
    elif (
        delta.count(0) == num_classes - 2
        and delta.count(1) == 1
        and delta.count(-1) == 1
    ):
        leaving_clss = delta.index(-1)
        arriving_clss = delta.index(1)
        number_waiting = state1[leaving_clss] - min(
            num_servers - min(num_servers, sum(state1[:leaving_clss])),
            state1[leaving_clss],
        )
        return number_waiting * thetas[leaving_clss][arriving_clss]
    return 0.0


def get_numbers_in_service(state, leaving_clss, num_servers, num_classes):
    """
    Returns the number of customerss of `leaving_clss` currently in service.
    """
    n = state[-1]
    if leaving_clss <= n:
        return min(num_servers, sum(state[: leaving_clss + 1])) - min(
            num_servers, sum(state[:leaving_clss])
        )
    if leaving_clss > n and leaving_clss != num_classes:
        return min(
            num_servers, sum(state[: leaving_clss + 1]) + 1 + state[num_classes]
        ) - min(num_servers, sum(state[:leaving_clss]) + 1 + state[num_classes])
    if leaving_clss == num_classes:
        return min(num_servers, sum(state[: n + 1]) + 1 + state[num_classes]) - min(
            num_servers, sum(state[: n + 1]) + 1
        )


def find_transition_rates_for_sojourn_time(
    state1, state2, num_servers, arrival_rates, service_rates, thetas
):
    """
    Finds the transition rates for given state transition
    """
    num_classes = len(arrival_rates)
    if state1 == "*":
        return 0
    n = state1[-1]
    if state2 == "*":
        if (
            sum(state1[: n + 1]) < num_servers
        ):  # I'm currently in service, when will I end service?
            return service_rates[n]
        else:
            return 0
    delta = tuple(state2[i] - state1[i] for i in range(num_classes + 2))
    if delta.count(0) == num_classes + 1:
        if delta.count(-1) == 1:
            leaving_clss = delta.index(-1)
            number_in_service = get_numbers_in_service(
                state1, leaving_clss, num_servers, num_classes
            )
            if (
                leaving_clss < num_classes
            ):  # Someone finishes service and leaves (not my class and behind me)
                return service_rates[leaving_clss] * number_in_service
            if (
                leaving_clss == num_classes
            ):  # Someone of my class behind me finishes service and leaves
                return service_rates[n] * number_in_service
        if delta.count(1) == 1:
            arriving_clss = delta.index(1)
            if (
                arriving_clss != n and arriving_clss < num_classes
            ):  # Customer not of my class arrives
                return arrival_rates[arriving_clss]
            if arriving_clss == num_classes:  # Customer of my class arrives
                return arrival_rates[n]
    if delta.count(0) == num_classes and delta.count(1) == 1 and delta.count(-1) == 1:
        leaving_clss = delta.index(-1)
        arriving_clss = delta.index(1)
        number_waiting = state1[leaving_clss] - get_numbers_in_service(
            state1, leaving_clss, num_servers, num_classes
        )
        if leaving_clss <= n and arriving_clss not in [
            n,
            num_classes,
            num_classes + 1,
        ]:  # Customer before me changing class not to my class
            return number_waiting * thetas[leaving_clss][arriving_clss]
        if (
            leaving_clss > n
            and leaving_clss < num_classes
            and arriving_clss not in [n, num_classes, num_classes + 1]
        ):  # Customer behind me (not my class) changing class not to my class
            return number_waiting * thetas[leaving_clss][arriving_clss]
        if leaving_clss == num_classes and arriving_clss not in [
            n,
            num_classes,
            num_classes + 1,
        ]:  # Customer behind me (of my class) changing class not to my class
            return number_waiting * thetas[n][arriving_clss]
        if (
            leaving_clss < n and arriving_clss == num_classes
        ):  # Customer before me changing class to my class
            return number_waiting * thetas[leaving_clss][n]
        if (
            leaving_clss > n
            and leaving_clss < num_classes
            and arriving_clss == num_classes
        ):  # Customer behind me (not my class) changing class to my class
            return number_waiting * thetas[leaving_clss][n]
    if (
        delta.count(0) == num_classes - 1
        and state1[num_classes] != 0
        and delta[n] == state1[num_classes]
        and delta[num_classes] == -state1[num_classes]
        and delta[-1] != 0
    ):  # I change class, there are people of my class behind me
        if sum(state1[: n + 1]) >= num_servers:  # I'm currently not in service
            new_n = state2[-1]
            return thetas[n][new_n]
    if (
        all(d == 0 for d in delta[:-1]) and state1[num_classes] == 0 and delta[-1] != 0
    ):  # I change class, no people of my class behind me
        if sum(state1[: n + 1]) >= num_servers:  # I'm currently not in service
            new_n = state2[-1]
            return thetas[n][new_n]
    return 0


def get_all_pairs_of_non_zero_entries_states(State_Space, bound):
    """
    Returns a list of all pairs of states that have a possible non-zero rate
    for the state probabilities markov chain.
    """
    all_pairs_indices = []
    for index, state1 in enumerate(State_Space):
        for i, s_i in enumerate(state1):
            state_arrival, state_service = list(state1), list(state1)
            if s_i < bound - 1:
                state_arrival[i] = s_i + 1
                next_state = tuple(state_arrival)
                all_pairs_indices.append((index, State_Space.index(next_state)))
            if s_i > 0:
                state_service[i] = s_i - 1
                next_state = tuple(state_service)
                all_pairs_indices.append((index, State_Space.index(next_state)))
            for j, s_j in enumerate(state1):
                state_change = list(state1)
                if i != j and s_j < bound - 1 and s_i > 0:
                    state_change[i] -= 1
                    state_change[j] += 1
                    next_state = tuple(state_change)
                    all_pairs_indices.append((index, State_Space.index(next_state)))
    return all_pairs_indices


def get_all_pairs_of_non_zero_entries_sojourn(State_Space, bound):
    """
    Returns a list of all pairs of states that have a possible non-zero rate
    for the sojourn time markov chain
    """
    all_pairs_indices = []
    num_classes = len(State_Space[0]) - 2
    for index, state1 in enumerate(State_Space[:-1]):
        n = state1[-1]
        b = state1[-2]

        # Go from any state to asterisk (I finish service)
        all_pairs_indices.append((index, len(State_Space) - 1))

        for i, s_i in enumerate(state1):
            if i not in (n, num_classes + 1) and s_i < bound - 1:
                # Arrival
                state_arrival = list(state1)
                state_arrival[i] = s_i + 1
                next_state = tuple(state_arrival)
                all_pairs_indices.append((index, State_Space.index(next_state)))
            if i < num_classes + 1 and s_i > 0:
                # Service
                state_service = list(state1)
                state_service[i] = s_i - 1
                next_state = tuple(state_service)
                all_pairs_indices.append((index, State_Space.index(next_state)))
            for j, s_j in enumerate(state1[:-1]):
                # Not me changing class
                if (
                    i != j
                    and i != (num_classes + 1)
                    and j != n
                    and s_i > 0
                    and s_j < bound - 1
                ):
                    if not (i == n and j == num_classes):
                        state_change = list(state1)
                        state_change[i] = s_i - 1
                        state_change[j] = s_j + 1
                        next_state = tuple(state_change)
                        all_pairs_indices.append((index, State_Space.index(next_state)))
        if state1[n] + b < bound:
            for new_n in range(num_classes):
                if new_n != n:
                    state_I_change = list(state1)
                    state_I_change[-2] = 0
                    state_I_change[n] = state1[n] + b
                    state_I_change[-1] = new_n
                    next_state = tuple(state_I_change)
                    all_pairs_indices.append((index, State_Space.index(next_state)))
    return all_pairs_indices


def write_transition_matrix(
    State_Space,
    transition_function,
    non_zero_pair_function,
    num_servers,
    arrival_rates,
    service_rates,
    thetas,
    bound,
):
    """
    Writes the transition matrix for the markov chain
    """
    size_mat = len(State_Space)
    transition_matrix = np.zeros((size_mat, size_mat))
    all_pairs = non_zero_pair_function(State_Space=State_Space, bound=bound)

    for s1, s2 in all_pairs:
        transition_matrix[s1, s2] = transition_function(
            state1=State_Space[s1],
            state2=State_Space[s2],
            num_servers=num_servers,
            arrival_rates=arrival_rates,
            service_rates=service_rates,
            thetas=thetas,
        )
    row_sums = np.sum(transition_matrix, axis=1)
    transition_matrix = transition_matrix - np.multiply(np.identity(size_mat), row_sums)
    return transition_matrix


def solve_probabilities(State_Space, transition_matrix):
    """
    Solves the steady state probabilities for the markov chain.
    """
    size_mat = len(State_Space)
    A = np.vstack((transition_matrix.transpose()[:-1], np.ones(size_mat)))
    b = np.vstack((np.zeros((size_mat - 1, 1)), [1]))
    sol = np.linalg.solve(A, b).transpose()[0]
    probs = {State_Space[i]: sol[i] for i in range(size_mat)}
    return probs


def find_mean_sojourn_time_by_class(num_classes, mean_sojourn_times, probs):
    """
    Finds the mean sojourn time by customer class.
    """
    mean_sojourn_times_by_class = {}
    for clss in range(num_classes):
        arriving_states = [
            s for s in mean_sojourn_times if s[-1] == clss and s[-2] == 0
        ]
        mean_sojourn_times_by_class[clss] = sum(
            [probs[state[:-2]] * mean_sojourn_times[state] for state in arriving_states]
        )
    return [mean_sojourn_times_by_class[clss] for clss in range(num_classes)]


def solve_time_to_absorbtion(State_Space, transition_matrix):
    """
    Finds the mean time to absorbtion for transition matrix.
    """
    size_mat = len(State_Space)
    T = transition_matrix[:-1, :-1]
    b = np.ones(size_mat - 1)
    time2absorb = np.linalg.solve(-T, b)
    mean_sojourn_time = {State_Space[i]: float(t) for i, t in enumerate(time2absorb)}
    return mean_sojourn_time


def aggregate_states(probs):
    """
    Aggregates from individual states to overall numbers of customers
    """
    agg_probs = {}
    for state in probs.keys():
        agg_state = sum(state)
        if agg_state in agg_probs:
            agg_probs[agg_state] += probs[state]
        else:
            agg_probs[agg_state] = probs[state]
    return agg_probs


def aggregate_states_by_class(probs, num_classes):
    """
    Aggregates from individual states to overall numbers of customers of each class
    """
    aggregate_probs_by_class = {clss: {} for clss in range(num_classes)}
    for clss in range(num_classes):
        for state in probs.keys():
            agg_state = state[clss]
            if agg_state in aggregate_probs_by_class[clss]:
                aggregate_probs_by_class[clss][agg_state] += probs[state]
            else:
                aggregate_probs_by_class[clss][agg_state] = probs[state]
    return aggregate_probs_by_class


def build_and_run_simulation(
    num_classes,
    num_servers,
    arrival_rates,
    service_rates,
    class_change_rate_matrix,
    max_simulation_time,
):
    """
    Builds and runs the simulation. Returns the simulation object after run.
    """
    N = ciw.create_network(
        arrival_distributions={
            "Class " + str(c): [ciw.dists.Exponential(arrival_rates[c])]
            for c in range(num_classes)
        },
        service_distributions={
            "Class " + str(c): [ciw.dists.Exponential(service_rates[c])]
            for c in range(num_classes)
        },
        number_of_servers=[num_servers],
        class_change_time_distributions=[
            [ciw.dists.Exponential(rate) if rate is not None else None for rate in row]
            for row in class_change_rate_matrix
        ],
        priority_classes=(
            {"Class " + str(c): c for c in range(num_classes)},
            ["resample"],
        ),
    )
    ciw.seed(0)
    Q = ciw.Simulation(N, tracker=ciw.trackers.NodeClassMatrix())
    Q.max_simulation_time = max_simulation_time
    Q.simulate_until_max_time(max_simulation_time, progress_bar=True)
    return Q


def get_state_probabilities_from_simulation(Q, warmup, cooldown):
    """
    Get state probabilities from Q state tracker.
    """
    obs_period = (warmup, Q.max_simulation_time - cooldown)
    probs = {
        state[0]: prob
        for state, prob in Q.statetracker.state_probabilities(
            observation_period=obs_period
        ).items()
    }
    return probs


def find_mean_sojourn_time_by_class_from_simulation(Q, num_classes, warmup):
    """
    Finds the mean sojourn time by customer class.
    """
    recs = Q.get_all_records()
    recs = [r for r in recs if r.arrival_date > warmup and r.record_type == "service"]
    mean_sojourn_times_by_class = {}
    for clss in range(num_classes):
        mean_sojourn_times_by_class[clss] = np.mean(
            [
                r.waiting_time + r.service_time
                for r in recs
                if r.original_customer_class == clss
            ]
        )
    return [mean_sojourn_times_by_class[clss] for clss in range(num_classes)]


def compare_mc_to_sim_states(
    num_classes,
    num_servers,
    arrival_rates,
    service_rates,
    thetas,
    bound,
    max_simulation_time,
    warmup,
    max_state,
):
    probs_mc = get_state_probabilities(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )
    agg_probs_mc = aggregate_states(probs_mc)
    agg_probs_by_class_mc = aggregate_states_by_class(probs_mc, num_classes)

    Q = build_and_run_simulation(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        class_change_rate_matrix=thetas,
        max_simulation_time=max_simulation_time,
    )
    probs_sim = get_state_probabilities_from_simulation(Q=Q, warmup=warmup)
    agg_probs_sim = aggregate_sim_states(probs_sim)
    agg_probs_by_class_sim = aggregate_sim_states_by_class(probs_sim, num_classes)

    fig, axarr = plt.subplots(1, 3, figsize=(17, 5))
    states = range(max_state)
    axarr[0].plot(
        states, [agg_probs_mc.get(s, 0) for s in states], label="Markov Chain"
    )
    axarr[0].plot(states, [agg_probs_sim.get(s, 0) for s in states], label="Simulation")
    axarr[0].set_xlabel("Number of Customers", fontsize=14)
    axarr[0].set_ylabel("Probability", fontsize=14)
    axarr[0].legend(fontsize=14, frameon=True)

    for clss in range(num_classes):
        axarr[1].plot(
            states,
            [agg_probs_by_class_mc[clss].get(s, 0) for s in states],
            label=f"Class {clss}",
        )
    axarr[1].set_title("Markov Chain", fontsize=16)
    axarr[1].legend(fontsize=14, frameon=True)
    axarr[1].set_xlabel("Number of Customers", fontsize=14)
    axarr[1].set_ylabel("Probability", fontsize=14)

    for clss in range(num_classes):
        axarr[2].plot(
            states,
            [agg_probs_by_class_sim[clss].get(s, 0) for s in states],
            label=f"Class {clss}",
        )
    axarr[2].set_title("Simulation", fontsize=16)
    axarr[2].legend(fontsize=14, frameon=True)
    axarr[2].set_xlabel("Number of Customers", fontsize=14)
    axarr[2].set_ylabel("Probability", fontsize=14)
    return fig


def compare_mc_to_sim_sojourn(
    num_classes,
    num_servers,
    arrival_rates,
    service_rates,
    thetas,
    bound,
    max_simulation_time,
    warmup,
):
    probs = get_state_probabilities(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )
    mean_sojourn_times_mc = get_mean_sojourn_times(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
        probs=probs,
    )
    Q = build_and_run_simulation(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        class_change_rate_matrix=thetas,
        max_simulation_time=max_simulation_time,
    )
    mean_sojourn_times_sim = find_mean_sojourn_time_by_class_from_simulation(
        Q, num_classes, warmup
    )
    return {"Markov Chain": mean_sojourn_times_mc, "Simulation": mean_sojourn_times_sim}


def find_hitting_probs(state_space, transition_matrix, boundary_region):
    """
    Finds the maximum probability of ever reaching a state in the boundary_region
    from every transient state not in the boundary_region.
    """
    P = transition_matrix.copy()
    n_states = len(state_space)
    Bi = [state_space.index(state) for state in boundary_region]
    absorbing_states = [si for si, s in enumerate(state_space) if P[si, si] == 1.0]
    transient_states = [
        si for si, s in enumerate(state_space) if si not in absorbing_states
    ]
    Bit = [transient_states.index(i) for i in Bi]
    n_transient = len(transient_states)
    for i in Bi:
        P[i] = np.zeros(n_states)
    A = P[:, transient_states][transient_states] - np.identity(n_transient)
    b = np.zeros(n_transient)
    for it in Bit:
        b[it] = -1
    p = np.linalg.solve(A, b)
    hitting_probabilities = {
        state_space[s]: p[si] for si, s in enumerate(transient_states)
    }
    return hitting_probabilities


def get_maximum_hitting_probs(
    state_space, transition_matrix, boundary, reasonable_ratio
):
    """
    Gets the maximum hitting probability within the reasonable region
    """
    boundary_region = [
        state for state in state_space if state != "*" if (boundary - 1) in state
    ]
    hitting_probs = find_hitting_probs(
        state_space=state_space,
        transition_matrix=transition_matrix,
        boundary_region=boundary_region,
    )
    reasonable_region = [
        state
        for state in state_space
        if state != "*"
        if sum(s**2 for s in state[:-1]) ** (1 / 2) <= (boundary * reasonable_ratio)
    ]
    return max([hitting_probs[state] for state in reasonable_region])


def bound_check(
    state_space,
    transition_matrix,
    boundary,
    reasonable_ratio,
    epsilon,
    max_hitting_prob=None,
):
    """
    Bound check for absorbing Markov chains. Checks wherther the maximum hitting
    probability for within the reasonable region to the boundary region is less
    than epsilon. This assumes that the Markov chain has a specific structure,
    where the head is the absorbing state and all other states go further away
    from the absorbing state as the elements of the states increase towards the
    boundary.

    Parameters
    ----------
    state_space : list
        List of states where each state is a tuple of numbers and the absorbing
        state is the first element of the list. The last number is a label
        representing what state you are currently in and does not contribute
        towards the distance from the absorbing state.
    transition_matrix : np.array
        Transition matrix for the state space.
    boundary : int
        The largest element of any state.
    reasonable_ratio : float [0, 1]
        The ratio of the of the state space that is considered in the reasonable
        region.
    epsilon : float > 0
        The tolerance for which the condition is satisfied.

    Returns
    -------
    bool
        True if the condition is satisfied.
    """
    if max_hitting_prob is None:
        max_hitting_prob = get_maximum_hitting_probs(
            state_space, transition_matrix, boundary, reasonable_ratio
        )
    condition = max_hitting_prob < epsilon
    return condition


def get_mean_sojourn_times(
    state_space_sojourn, transition_matrix_sojourn, num_classes, arrival_rates, probs
):
    """
    Get the mean sojourn times for the given state probabilities
    """
    mean_sojourn_times = solve_time_to_absorbtion(
        State_Space=state_space_sojourn, transition_matrix=transition_matrix_sojourn
    )
    mean_sojourn_times_by_class = find_mean_sojourn_time_by_class(
        num_classes=num_classes, mean_sojourn_times=mean_sojourn_times, probs=probs
    )
    overall_sojourn_time = sum(
        [
            sojourn_time * arr_rate
            for sojourn_time, arr_rate in zip(
                arrival_rates, mean_sojourn_times_by_class
            )
        ]
    ) / sum(arrival_rates)

    return mean_sojourn_times_by_class + [overall_sojourn_time]


def get_average_num_of_customers_from_state_probs(state_probs, num_classes):
    """
    Gets the average number of customers in the system for all classes
    of customers and the combined average for all classes
    """
    average_num_of_customers = [0 for _ in range(num_classes + 1)]
    for state, prob in state_probs.items():
        for class_id in range(num_classes):
            average_num_of_customers[class_id] += prob * state[class_id]
        average_num_of_customers[-1] += prob * sum(state)
    return average_num_of_customers


def get_variance_of_number_of_customers_from_state_probs(
    state_probs, average_in_system, num_classes
):
    """
    Gets the variance of the number of customers in the system for
    all classes of customers and the combined average for all classes
    """
    variance_of_num_of_customers = [0 for _ in range(num_classes + 1)]
    for state, prob in state_probs.items():
        for class_id in range(num_classes):
            variance_of_num_of_customers[class_id] += (
                prob * (state[class_id] - average_in_system[class_id]) ** 2
            )
        variance_of_num_of_customers[-1] += (
            prob * (sum(state) - average_in_system[-1]) ** 2
        )
    return variance_of_num_of_customers


def get_average_num_of_customers_waiting_from_state_probs(
    state_probs, num_servers, num_classes
):
    """
    Get the average number of waiting customers
    """
    average_num_waiting = [0 for _ in range(num_classes + 1)]
    for state, prob in state_probs.items():
        for class_id in range(num_classes):
            average_num_waiting[class_id] += prob * max(
                state[class_id] - num_servers + min(sum(state[:class_id]), num_servers),
                0,
            )
        average_num_waiting[-1] += prob * max(sum(state) - num_servers, 0)
    return average_num_waiting


def get_variance_of_customers_waiting_from_state_probs(
    state_probs, num_servers, average_waiting, num_classes
):
    """
    Get the variance of the number of waiting customers
    """
    average_num_waiting = [0 for _ in range(num_classes + 1)]
    for state, prob in state_probs.items():
        for class_id in range(num_classes):
            average_num_waiting[class_id] += (
                prob
                * (
                    max(
                        state[class_id]
                        - num_servers
                        + min(sum(state[:class_id]), num_servers),
                        0,
                    )
                    - average_waiting[class_id]
                )
                ** 2
            )
        average_num_waiting[-1] += (
            prob * (max(sum(state) - num_servers, 0) - average_waiting[-1]) ** 2
        )
    return average_num_waiting


def get_empty_probabilities_from_state_probs(state_probs, num_classes):
    """
    Gets the probabilities of no customers of all specific class and
    the probability of no customers at all
    """
    empty_probs = [0 for _ in range(num_classes + 1)]
    empty_probs[-1] = state_probs[tuple(0 for _ in range(num_classes))]
    for state, prob in state_probs.items():
        for class_id in range(num_classes):
            if state[class_id] == 0:
                empty_probs[class_id] += prob
    return empty_probs


def get_mean_sojourn_times_using_simulation(
    Q, max_simulation_time, warmup_time, cooldown_time, num_classes
):
    """
    Simulate the system and get the mean sojourn times for each class
    """
    mean_sojourn_time_by_class = [
        np.mean(
            [
                q.waiting_time + q.service_time
                for q in Q.get_all_records()
                if q.customer_class == class_id
                and q.arrival_date > warmup_time
                and q.arrival_date < max_simulation_time - cooldown_time
            ]
        )
        for class_id in range(num_classes)
    ]

    overall_mean_sojour_time = np.mean(
        [
            q.waiting_time + q.service_time
            for q in Q.get_all_records()
            if q.arrival_date > warmup_time
            and q.arrival_date < max_simulation_time - cooldown_time
        ]
    )

    return mean_sojourn_time_by_class + [overall_mean_sojour_time]


def get_simulation_performance_measures(
    Q, num_servers, num_classes, max_simulation_time, warmup_time, cooldown_time
):
    """
    Get all performance measures using the simulation
    """
    state_probs_sim = get_state_probabilities_from_simulation(
        Q=Q, warmup=warmup_time, cooldown=cooldown_time
    )
    mean_custs = get_average_num_of_customers_from_state_probs(
        state_probs=state_probs_sim, num_classes=num_classes
    )
    variance_custs = get_variance_of_number_of_customers_from_state_probs(
        state_probs=state_probs_sim,
        average_in_system=mean_custs,
        num_classes=num_classes,
    )
    mean_waiting = get_average_num_of_customers_waiting_from_state_probs(
        state_probs=state_probs_sim, num_servers=num_servers, num_classes=num_classes
    )
    variance_waiting = get_variance_of_customers_waiting_from_state_probs(
        state_probs=state_probs_sim,
        num_servers=num_servers,
        average_waiting=mean_waiting,
        num_classes=num_classes,
    )
    empty_probs = get_empty_probabilities_from_state_probs(
        state_probs=state_probs_sim, num_classes=num_classes
    )
    mean_sojourn_times = get_mean_sojourn_times_using_simulation(
        Q=Q,
        max_simulation_time=max_simulation_time,
        warmup_time=warmup_time,
        cooldown_time=cooldown_time,
        num_classes=num_classes,
    )

    return (
        mean_custs,
        variance_custs,
        mean_waiting,
        variance_waiting,
        empty_probs,
        mean_sojourn_times,
    )


def get_markov_perfromance_measures(
    arrival_rates,
    state_probs,
    state_space_sojourn,
    transition_matrix_sojourn,
    num_classes,
    num_servers,
    bound,
):
    """
    Get all performance measures using the Markov chain
    """
    mean_custs = get_average_num_of_customers_from_state_probs(
        state_probs=state_probs, num_classes=num_classes
    )
    variance_custs = get_variance_of_number_of_customers_from_state_probs(
        state_probs=state_probs,
        average_in_system=mean_custs,
        num_classes=num_classes,
    )
    mean_waiting = get_average_num_of_customers_waiting_from_state_probs(
        state_probs=state_probs,
        num_servers=num_servers,
        num_classes=num_classes,
    )
    variance_waiting = get_variance_of_customers_waiting_from_state_probs(
        state_probs=state_probs,
        num_servers=num_servers,
        average_waiting=mean_waiting,
        num_classes=num_classes,
    )
    empty_probs = get_empty_probabilities_from_state_probs(
        state_probs=state_probs, num_classes=num_classes
    )
    mean_sojourn_times = get_mean_sojourn_times(
        state_space_sojourn=state_space_sojourn,
        transition_matrix_sojourn=transition_matrix_sojourn,
        num_classes=num_classes,
        arrival_rates=arrival_rates,
        probs=state_probs,
    )

    return (
        mean_custs,
        variance_custs,
        mean_waiting,
        variance_waiting,
        empty_probs,
        mean_sojourn_times,
    )


def write_row(
    num_classes,
    arrival_rates,
    service_rates,
    num_servers,
    thetas,
    bound,
    reasonable_ratio,
    epsilon,
    max_simulation_time,
    warmup_time,
    cooldown_time,
    force_simulation=False,
):
    """
    Get a row of the results that will be written to the csv file. The function
    first checks if force_simulation is set to True and runs the simulation if
    so. Otherwise, it checks if the bound is big enough to run the Markov chain.
    If not, it runs the simulation. Otherwise, it runs the Markov chain. 
    The generated row consists of the following values:

        Simulation time,
        Warmup time,
        Cooldown time,
        Boundary,
        Reasonable ratio,
        Epsilon,
        Number of classes,
        Arrival rates,
        Service rates,
        Thetas,
        Number of servers,
        Mean number of customers in the system,
        Variance of the number of customers in the system,
        Mean number of customers waiting,
        Variance of the number of customers waiting,
        Probability of the system being empty,
        Mean sojourn time,
        Maximum hitting probability

    Note that when the simulation is used Bound, Reasonable ratio and Epsilon are
    set to None. In addtion when the Markov chain is used, Simulation time, Warmup
    time and Cooldown time are set to None.
    """
    if force_simulation:
        Q = build_and_run_simulation(
            num_classes=num_classes,
            num_servers=num_servers,
            arrival_rates=arrival_rates,
            service_rates=service_rates,
            class_change_rate_matrix=thetas,
            max_simulation_time=max_simulation_time,
        )

        (
            mean_custs,
            variance_custs,
            mean_waiting,
            variance_waiting,
            empty_probs,
            mean_sojourn_times,
        ) = get_simulation_performance_measures(
            Q=Q,
            num_servers=num_servers,
            num_classes=num_classes,
            max_simulation_time=max_simulation_time,
            warmup_time=warmup_time,
            cooldown_time=cooldown_time,
        )

        bound, reasonable_ratio, epsilon, max_hitting_probs = None, None, None, None

    else:

        state_probs = get_state_probabilities(
            num_classes=num_classes,
            num_servers=num_servers,
            arrival_rates=arrival_rates,
            service_rates=service_rates,
            thetas=thetas,
            bound=bound,
        )

        (
            state_space_sojourn,
            transition_matrix_sojourn,
        ) = build_state_space_and_transition_matrix_sojourn_mc(
            num_classes=num_classes,
            num_servers=num_servers,
            arrival_rates=arrival_rates,
            service_rates=service_rates,
            thetas=thetas,
            bound=bound,
        )
        max_hitting_probs = get_maximum_hitting_probs(
            state_space=state_space_sojourn,
            transition_matrix=transition_matrix_sojourn,
            boundary=bound,
            reasonable_ratio=reasonable_ratio,
        )
        use_markov = bound_check(
            state_space=None,
            transition_matrix=None,
            boundary=None,
            reasonable_ratio=None,
            epsilon=epsilon,
            max_hitting_prob=max_hitting_probs,
        )

        if use_markov:
            (
                mean_custs,
                variance_custs,
                mean_waiting,
                variance_waiting,
                empty_probs,
                mean_sojourn_times,
            ) = get_markov_perfromance_measures(
                arrival_rates=arrival_rates,
                state_probs=state_probs,
                state_space_sojourn=state_space_sojourn,
                transition_matrix_sojourn=transition_matrix_sojourn,
                num_classes=num_classes,
                num_servers=num_servers,
                bound=bound,
            )

            max_simulation_time, warmup_time, cooldown_time = None, None, None

        else:
            # I thinks this should not be here. It does not make sense to run
            # the simulation if the Markov chain is not used. We don't know if
            # we checked all bounds yet.
            Q = build_and_run_simulation(
                num_classes=num_classes,
                num_servers=num_servers,
                arrival_rates=arrival_rates,
                service_rates=service_rates,
                class_change_rate_matrix=thetas,
                max_simulation_time=max_simulation_time,
            )

            (
                mean_custs,
                variance_custs,
                mean_waiting,
                variance_waiting,
                empty_probs,
                mean_sojourn_times,
            ) = get_simulation_performance_measures(
                Q=Q,
                num_servers=num_servers,
                num_classes=num_classes,
                max_simulation_time=max_simulation_time,
                warmup_time=warmup_time,
                cooldown_time=cooldown_time,
            )

            bound, reasonable_ratio, epsilon, max_hitting_probs = None, None, None, None

    return (
        max_simulation_time,
        warmup_time,
        cooldown_time,
        bound,
        reasonable_ratio,
        epsilon,
        num_classes,
        arrival_rates,
        service_rates,
        thetas,
        num_servers,
        mean_custs,
        variance_custs,
        mean_waiting,
        variance_waiting,
        empty_probs,
        mean_sojourn_times,
        max_hitting_probs,
    )