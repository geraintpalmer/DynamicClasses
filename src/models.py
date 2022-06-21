import numpy as np
import itertools
import ciw
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")


def get_state_probabilities(
    num_classes, num_servers, arrival_rates, service_rates, thetas, infty
):
    """
    Get's the system's state probabilities
    """
    State_Space = write_state_space_for_states(num_classes=num_classes, infty=infty)
    transition_matrix = write_transition_matrix(
        State_Space=State_Space,
        transition_function=find_transition_rates_for_states,
        non_zero_pair_function=get_all_pairs_of_non_zero_entries_states,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        infty=infty,
    )
    probs = solve_probabilities(
        State_Space=State_Space, transition_matrix=transition_matrix
    )
    return probs


def get_mean_sojourn_times(
    num_classes, num_servers, arrival_rates, service_rates, thetas, infty, probs
):
    """
    Get the mean sojourn times for the given state probabilities
    """
    State_Space = write_state_space_for_sojourn(num_classes=num_classes, infty=infty)
    transition_matrix = write_transition_matrix(
        State_Space=State_Space,
        transition_function=find_transition_rates_for_sojourn_time,
        non_zero_pair_function=get_all_pairs_of_non_zero_entries_sojourn,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        infty=infty,
    )
    mean_sojourn_times = solve_time_to_absorbtion(
        State_Space=State_Space, transition_matrix=transition_matrix
    )
    mean_sojourn_times_by_class = find_mean_sojourn_time_by_class(
        num_classes=num_classes, mean_sojourn_times=mean_sojourn_times, probs=probs
    )
    return mean_sojourn_times_by_class


def write_state_space_for_states(num_classes, infty):
    """
    Write the states for the state probability model
    """
    State_Space = list(itertools.product(*[range(infty) for _ in range(num_classes)]))
    return State_Space


def write_state_space_for_sojourn(num_classes, infty):
    """
    Write the states for the sojourn time model
    """
    State_Space = list(
        itertools.product(
            *(
                [range(infty) for _ in range(num_classes)]
                + [range(infty)]
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


def get_all_pairs_of_non_zero_entries_states(State_Space, infty):
    """
    Returns a list of all pairs of states that have a possible non-zero rate
    for the state probabilities markov chain.
    """
    all_pairs_indices = []
    for index, state1 in enumerate(State_Space):
        for i, s_i in enumerate(state1):
            state_arrival, state_service = list(state1), list(state1)
            if s_i < infty - 1:
                state_arrival[i] = s_i + 1
                next_state = tuple(state_arrival)
                all_pairs_indices.append((index, State_Space.index(next_state)))
            if s_i > 0:
                state_service[i] = s_i - 1
                next_state = tuple(state_service)
                all_pairs_indices.append((index, State_Space.index(next_state)))
            for j, s_j in enumerate(state1):
                state_change = list(state1)
                if i != j and s_j < infty - 1 and s_i > 0:
                    state_change[i] -= 1
                    state_change[j] += 1
                    next_state = tuple(state_change)
                    all_pairs_indices.append((index, State_Space.index(next_state)))
    return all_pairs_indices


def get_all_pairs_of_non_zero_entries_sojourn(State_Space, infty):
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
            if i not in (n, num_classes + 1) and s_i < infty - 1:
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
                    and s_j < infty - 1
                ):
                    if not (i == n and j == num_classes):
                        state_change = list(state1)
                        state_change[i] = s_i - 1
                        state_change[j] = s_j + 1
                        next_state = tuple(state_change)
                        all_pairs_indices.append((index, State_Space.index(next_state)))
        if state1[n] + b < infty:
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
    infty,
):
    """
    Writes the transition matrix for the markov chain
    """
    size_mat = len(State_Space)
    transition_matrix = np.zeros((size_mat, size_mat))
    all_pairs = non_zero_pair_function(State_Space=State_Space, infty=infty)

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


def aggregate_mc_states(probs):
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


def aggregate_mc_states_by_class(probs, num_classes):
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


def aggregate_sim_states(probs):
    """
    Aggregates from individual states to overall numbers of customers
    """
    agg_probs = {}
    for state in probs.keys():
        agg_state = sum(state[0])
        if agg_state in agg_probs:
            agg_probs[agg_state] += probs[state]
        else:
            agg_probs[agg_state] = probs[state]
    return agg_probs


def aggregate_sim_states_by_class(probs, num_classes):
    """
    Aggregates from individual states to overall numbers of customers of each class
    """
    aggregate_probs_by_class = {clss: {} for clss in range(num_classes)}
    for clss in range(num_classes):
        for state in probs.keys():
            agg_state = state[0][clss]
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
    max_simulation_time
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


def get_state_probabilities_from_simulation(Q, warmup):
    """
    Get state probabilities from Q state tracker.
    """
    obs_period = (warmup, Q.max_simulation_time - warmup)
    probs = Q.statetracker.state_probabilities(observation_period=obs_period)
    return probs


def find_mean_sojourn_time_by_class_from_simulation(Q, num_classes, warmup):
    """
    Finds the mean sojourn time by customer class.
    """
    recs = Q.get_all_records()
    recs = [r for r in recs if r.arrival_date > warmup and r.record_type == 'service']
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
    infty,
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
        infty=infty,
    )
    agg_probs_mc = aggregate_mc_states(probs_mc)
    agg_probs_by_class_mc = aggregate_mc_states_by_class(probs_mc, num_classes)

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
    infty,
    max_simulation_time,
    warmup,
):
    probs = get_state_probabilities(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        infty=infty,
    )
    mean_sojourn_times_mc = get_mean_sojourn_times(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        infty=infty,
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
