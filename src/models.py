import numpy as np
import itertools
import ciw


class MarkovChain:
    """
    A class to hold the Markov chain object for a multi-class dynamic classes priority MMc queue
    """

    def __init__(
        self,
        number_of_servers,
        arrival_rates,
        service_rates,
        class_change_rate_matrix,
        inf,
        sojourn=False,
    ):
        """
        Initialises the Markov Chain object
        """
        self.c = number_of_servers
        self.ls = arrival_rates
        self.ms = service_rates
        self.thetas = class_change_rate_matrix
        self.inf = inf
        self.k = len(arrival_rates)
        self.State_Space = list(
            itertools.product(*[range(self.inf) for _ in range(self.k)])
        )
        self.lenmat = len(self.State_Space)
        self.write_transition_matrix()
        self.discretise_transition_matrix()
        self.solve()
        self.aggregate_states()
        self.aggregate_states_by_class()
        if sojourn:
            self.SojournTime_MC = MarkovChain_SoujournTime(
                number_of_servers,
                arrival_rates,
                service_rates,
                class_change_rate_matrix,
                inf,
            )
            self.find_mean_sojourn_time_by_class()

    def find_transition_rates(self, state1, state2):
        """
        Finds the transition rates for given state transition
        """
        delta = tuple(state2[i] - state1[i] for i in range(self.k))
        if delta.count(0) == self.k - 1:
            if delta.count(-1) == 1:
                leaving_clss = delta.index(-1)
                if sum([state1[clss] for clss in range(leaving_clss)]) < self.c:
                    number_in_service = min(
                        self.c - min(self.c, sum(state1[:leaving_clss])),
                        state1[leaving_clss],
                    )
                    return self.ms[leaving_clss] * number_in_service
            elif delta.count(1) == 1:
                arriving_clss = delta.index(1)
                return self.ls[arriving_clss]
        elif (
            delta.count(0) == self.k - 2
            and delta.count(1) == 1
            and delta.count(-1) == 1
        ):
            leaving_clss = delta.index(-1)
            arriving_clss = delta.index(1)
            number_waiting = state1[leaving_clss] - min(
                self.c - min(self.c, sum(state1[:leaving_clss])), state1[leaving_clss]
            )
            return number_waiting * self.thetas[leaving_clss][arriving_clss]
        return 0.0

    def write_transition_matrix(self):
        """
        Writes the transition matrix for the markov chain
        """
        transition_matrix = np.array(
            [
                [self.find_transition_rates(s1, s2) for s2 in self.State_Space]
                for s1 in self.State_Space
            ]
        )
        row_sums = np.sum(transition_matrix, axis=1)
        self.time_step = 1 / np.max(row_sums)
        self.transition_matrix = transition_matrix - np.multiply(
            np.identity(self.lenmat), row_sums
        )

    def discretise_transition_matrix(self):
        """
        Disctetises the transition matrix
        """
        self.discrete_transition_matrix = (
            self.transition_matrix * self.time_step + np.identity(self.lenmat)
        )

    def solve(self):
        A = np.append(
            np.transpose(self.discrete_transition_matrix) - np.identity(self.lenmat),
            [[1 for _ in range(self.lenmat)]],
            axis=0,
        )
        b = np.transpose(np.array([0 for _ in range(self.lenmat)] + [1]))
        sol = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))
        self.probs = {self.State_Space[i]: sol[i] for i in range(self.lenmat)}

    def aggregate_states(self):
        """
        Aggregates from individual states to overall numbers of customers
        """
        agg_probs = {}
        for state in self.probs.keys():
            agg_state = sum(state)
            if agg_state in agg_probs:
                agg_probs[agg_state] += self.probs[state]
            else:
                agg_probs[agg_state] = self.probs[state]
        self.aggregate_probs = agg_probs

    def aggregate_states_by_class(self):
        """
        Aggregates from individual states to overall numbers of customers of each class
        """
        self.aggregate_probs_by_class = {clss: {} for clss in range(self.k)}
        for clss in range(self.k):
            for state in self.probs.keys():
                agg_state = state[clss]
                if agg_state in self.aggregate_probs_by_class[clss]:
                    self.aggregate_probs_by_class[clss][agg_state] += self.probs[state]
                else:
                    self.aggregate_probs_by_class[clss][agg_state] = self.probs[state]

    def find_mean_sojourn_time_by_class(self):
        """
        Finds the mean sojourn time by customer class.
        """
        self.mean_sojourn_times_by_class = {}
        for clss in range(self.k):
            arriving_states = [
                s
                for s in self.SojournTime_MC.mean_steps_to_absorbtion
                if s[-1] == clss and s[-2] == 0
            ]
            self.mean_sojourn_times_by_class[clss] = sum(
                [
                    self.probs[state[:-2]]
                    * self.SojournTime_MC.mean_sojourn_time[state]
                    for state in arriving_states
                ]
            )


class MarkovChain_SoujournTime:
    """
    A class to hold the abosprbin Markov chain object to find the sojourn time for a multi-class dynamic classes priority MMc queue
    """

    def __init__(
        self,
        number_of_servers,
        arrival_rates,
        service_rates,
        class_change_rate_matrix,
        inf,
    ):
        """
        Initialises the Markov Chain object
        """
        self.c = number_of_servers
        self.ls = arrival_rates
        self.ms = service_rates
        self.thetas = class_change_rate_matrix
        self.inf = inf
        self.k = len(arrival_rates)
        self.State_Space = list(
            itertools.product(
                *(
                    [range(self.inf) for _ in range(self.k)]
                    + [range(self.inf)]
                    + [range(self.k)]
                )
            )
        ) + ["*"]
        self.lenmat = len(self.State_Space)
        self.write_transition_matrix()
        self.discretise_transition_matrix()
        self.solve()

    def find_transition_rates(self, state1, state2):
        """
        Finds the transition rates for given state transition
        """
        if state1 == "*":
            return 0
        n = state1[-1]
        if state2 == "*":
            if (
                sum(state1[: n + 1]) < self.c
            ):  # I'm currently in service, when will I end service?
                return self.ms[n]
            else:
                return 0
        delta = tuple(state2[i] - state1[i] for i in range(self.k + 2))
        if delta.count(0) == self.k + 1:
            if delta.count(-1) == 1:
                leaving_clss = delta.index(-1)
                if (
                    leaving_clss <= n
                ):  # Someone in front of me finishes service and leaves
                    number_in_service = min(
                        self.c - min(self.c, sum(state1[:leaving_clss])),
                        state1[leaving_clss],
                    )
                    return self.ms[leaving_clss] * number_in_service
                if (
                    leaving_clss > n and leaving_clss < self.k
                ):  # Someone behind me finishes service and leaves
                    number_in_service = min(
                        self.c
                        - min(self.c, sum(state1[:leaving_clss]) + 1 + state1[self.k]),
                        state1[leaving_clss],
                    )
                    return self.ms[leaving_clss] * number_in_service
                if (
                    leaving_clss == self.k
                ):  # Someone of my class behind me finishes service and leaves
                    number_in_service = min(
                        self.c - min(self.c, sum(state1[: n + 1]) + 1), state1[self.k]
                    )
                    return self.ms[n] * number_in_service
            if delta.count(1) == 1:
                arriving_clss = delta.index(1)
                if (
                    arriving_clss != n and arriving_clss < self.k
                ):  # Customer not of my class arrives
                    return self.ls[arriving_clss]
                if arriving_clss == self.k:  # Customer of my class arrives
                    return self.ls[n]
        if delta.count(0) == self.k and delta.count(1) == 1 and delta.count(-1) == 1:
            leaving_clss = delta.index(-1)
            arriving_clss = delta.index(1)
            if leaving_clss <= n and arriving_clss not in [
                n,
                self.k,
                self.k + 1,
            ]:  # Customer before me changing class not to my class
                number_waiting = state1[leaving_clss] - min(
                    self.c - min(self.c, sum(state1[:leaving_clss])),
                    state1[leaving_clss],
                )
                return number_waiting * self.thetas[leaving_clss][arriving_clss]
            if (
                leaving_clss > n
                and leaving_clss < self.k
                and arriving_clss not in [n, self.k, self.k + 1]
            ):  # Customer behind me (not my class) changing class not to my class
                number_waiting = state1[leaving_clss] - min(
                    self.c
                    - min(self.c, sum(state1[:leaving_clss]) + 1 + state1[self.k]),
                    state1[leaving_clss],
                )
                return number_waiting * self.thetas[leaving_clss][arriving_clss]
            if leaving_clss == self.k and arriving_clss not in [
                n,
                self.k,
                self.k + 1,
            ]:  # Customer behind me (of my class) changing class not to my class
                number_waiting = state1[leaving_clss] - min(
                    self.c - min(self.c, sum(state1[: n + 1]) + 1), state1[leaving_clss]
                )
                return number_waiting * self.thetas[n][arriving_clss]
            if (
                leaving_clss < n and arriving_clss == self.k
            ):  # Customer before me changing class to my class
                number_waiting = state1[leaving_clss] - min(
                    self.c - min(self.c, sum(state1[:leaving_clss])),
                    state1[leaving_clss],
                )
                return number_waiting * self.thetas[leaving_clss][n]
            if (
                leaving_clss > n and leaving_clss < self.k and arriving_clss == self.k
            ):  # Customer behind me (not my class) changing class to my class
                number_waiting = state1[leaving_clss] - min(
                    self.c
                    - min(self.c, sum(state1[:leaving_clss]) + 1 + state1[self.k]),
                    state1[leaving_clss],
                )
                return number_waiting * self.thetas[leaving_clss][n]
        if (
            delta.count(0) == self.k - 1
            and state1[self.k] != 0
            and delta[n] == state1[self.k]
            and delta[self.k] == -state1[self.k]
            and delta[-1] != 0
        ):  # I change class, there are people of my class behind me
            if sum(state1[: n + 1]) >= self.c:  # I'm currently not in service
                new_n = state2[-1]
                return self.thetas[n][new_n]
        if (
            all(d == 0 for d in delta[:-1]) and state1[self.k] == 0 and delta[-1] != 0
        ):  # I change class, no people of my class behind me
            if sum(state1[: n + 1]) >= self.c:  # I'm currently not in service
                new_n = state2[-1]
                return self.thetas[n][new_n]
        return 0

    def write_transition_matrix(self):
        """
        Writes the transition matrix for the markov chain
        """
        transition_matrix = np.array(
            [
                [self.find_transition_rates(s1, s2) for s2 in self.State_Space]
                for s1 in self.State_Space
            ]
        )
        row_sums = np.sum(transition_matrix, axis=1)
        self.time_step = 1 / np.max(row_sums)
        self.transition_matrix = transition_matrix - np.multiply(
            np.identity(self.lenmat), row_sums
        )

    def discretise_transition_matrix(self):
        """
        Disctetises the transition matrix
        """
        self.discrete_transition_matrix = (
            self.transition_matrix * self.time_step + np.identity(self.lenmat)
        )

    def solve(self):
        """
        Finds the mean time to absorbtion for discretised transition matrix.
        """
        T = self.discrete_transition_matrix[:-1, :-1]
        S = np.linalg.inv(np.identity(len(T)) - T)
        steps2absorb = [sum([S[i, j] for j in range(len(S))]) for i in range(len(S))]
        time2absorb = [s * self.time_step for s in steps2absorb]
        self.mean_steps_to_absorbtion = {
            self.State_Space[i]: steps2absorb[i] for i in range(len(steps2absorb))
        }
        self.mean_sojourn_time = {
            self.State_Space[i]: float(time2absorb[i]) for i in range(len(time2absorb))
        }


class Simulation:
    """
    A class to hold the Simulation for a multi-class dynamic classes priority MMc queue
    """

    def __init__(
        self,
        number_of_servers,
        arrival_rates,
        service_rates,
        class_change_rate_matrix,
        preempt,
        max_simulation_time,
        warmup,
    ):
        self.max_simulation_time = max_simulation_time
        self.warmup = warmup
        self.obs_period = (self.warmup, self.max_simulation_time - self.warmup)
        self.k = len(arrival_rates)
        self.N = ciw.create_network(
            arrival_distributions={
                "Class " + str(c): [ciw.dists.Exponential(arrival_rates[c])]
                for c in range(self.k)
            },
            service_distributions={
                "Class " + str(c): [ciw.dists.Exponential(service_rates[c])]
                for c in range(self.k)
            },
            number_of_servers=[number_of_servers],
            class_change_time_distributions=[
                [
                    ciw.dists.Exponential(rate) if rate is not None else None
                    for rate in row
                ]
                for row in class_change_rate_matrix
            ],
            priority_classes=({"Class " + str(c): c for c in range(self.k)}, [preempt]),
        )
        self.run()
        self.aggregate_states()
        self.aggregate_states_by_class()
        self.find_mean_sojourn_time_by_class()

    def run(self):
        """
        Runs the simulation and finds the state probabilities
        """
        ciw.seed(0)
        self.Q = ciw.Simulation(self.N, tracker=ciw.trackers.NodeClassMatrix())
        self.Q.simulate_until_max_time(self.max_simulation_time, progress_bar=True)
        self.probs = self.Q.statetracker.state_probabilities(
            observation_period=self.obs_period
        )

    def aggregate_states(self):
        """
        Aggregates from individual states to overall numbers of customers
        """
        self.aggregate_probs = {}
        for state in self.probs.keys():
            agg_state = sum(state[0])
            if agg_state in self.aggregate_probs:
                self.aggregate_probs[agg_state] += self.probs[state]
            else:
                self.aggregate_probs[agg_state] = self.probs[state]

    def aggregate_states_by_class(self):
        """
        Aggregates from individual states to overall numbers of customers of each class
        """
        self.aggregate_probs_by_class = {clss: {} for clss in range(self.k)}
        for clss in range(self.k):
            for state in self.probs.keys():
                agg_state = state[0][clss]
                if agg_state in self.aggregate_probs_by_class[clss]:
                    self.aggregate_probs_by_class[clss][agg_state] += self.probs[state]
                else:
                    self.aggregate_probs_by_class[clss][agg_state] = self.probs[state]

    def find_mean_sojourn_time_by_class(self):
        """
        Finds the mean sojourn time by customer class.
        """
        recs = self.Q.get_all_records()
        self.mean_sojourn_times_by_class = {}
        for clss in range(self.k):
            self.mean_sojourn_times_by_class[clss] = np.mean(
                [
                    r.waiting_time + r.service_time
                    for r in recs
                    if r.original_customer_class == clss
                ]
            )
