import numpy as np
import itertools

# Import Ciw from source - required features not merged yet.
import sys
sys.path.append("../../")
import CiwPython.Ciw.ciw as ciw
import CiwPython.Ciw.ciw.dists as dists
import CiwPython.Ciw.ciw.trackers as trackers


class MarkovChain:
    """
    A class to hold the Markov chain object for a multi-class dynamic classes priority MMc queue
    """
    def __init__(self, number_of_servers, arrival_rates, service_rates, class_change_rate_matrix, inf):
        """
        Initialises the Markov Chain object
        """
        self.c = number_of_servers
        self.ls = arrival_rates
        self.ms = service_rates
        self.thetas = class_change_rate_matrix
        self.inf = inf
        self.k = len(arrival_rates)
        self.State_Space = list(itertools.product(*[range(self.inf) for _ in range(self.k)]))
        self.lenmat = len(self.State_Space)
        self.write_transition_matrix()
        self.discretise_transition_matrix()
        self.solve()
        self.aggregate_states()
        self.aggregate_states_by_class()

    def find_transition_rates(self, state1, state2):
        """
        Finds the transition rates for given state transition
        """
        delta = tuple(state2[i] - state1[i] for i in range(self.k))
        if delta.count(0) == self.k - 1:
            if delta.count(-1) == 1:
                leaving_clss = delta.index(-1)
                if sum([state1[clss] for clss in range(leaving_clss)]) < self.c:
                    number_in_service = min(self.c - min(self.c, sum(state1[:leaving_clss])), state1[leaving_clss])
                    return self.ms[leaving_clss] * number_in_service
            elif delta.count(1) == 1:
                arriving_clss = delta.index(1)
                return self.ls[arriving_clss]
        elif delta.count(0) == self.k - 2 and delta.count(1) == 1 and delta.count(-1) == 1:
            leaving_clss = delta.index(-1)
            arriving_clss = delta.index(1)
            number_waiting = state1[leaving_clss] - min(self.c - min(self.c, sum(state1[:leaving_clss])), state1[leaving_clss])
            return number_waiting * self.thetas[leaving_clss][arriving_clss]
        return 0.0

    def write_transition_matrix(self):
        """
        Writes the transition matrix for the markov chain
        """
        transition_matrix = np.array([[self.find_transition_rates(s1, s2) for s2 in self.State_Space] for s1 in self.State_Space])
        row_sums = np.sum(transition_matrix, axis=1)
        self.time_step = 1 / np.max(row_sums)
        self.transition_matrix = transition_matrix - np.multiply(np.identity(self.lenmat), row_sums)

    def discretise_transition_matrix(self):
        """
        Disctetises the transition matrix
        """
        self.discrete_transition_matrix = self.transition_matrix*self.time_step + np.identity(self.lenmat)

    def solve(self):
        A = np.append(np.transpose(self.discrete_transition_matrix) - np.identity(self.lenmat), [[1 for _ in range(self.lenmat)]], axis=0)
        b = np.transpose(np.array([0 for _ in range(self.lenmat)] + [1]))
        sol = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))
        self.probs =  {self.State_Space[i]: sol[i] for i in range(self.lenmat)}

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


class Simulation:
    """
    A class to hold the Simulation for a multi-class dynamic classes priority MMc queue
    """
    def __init__(self, number_of_servers, arrival_rates, service_rates, class_change_rate_matrix, preempt, max_simulation_time, warmup):
        self.max_simulation_time = max_simulation_time
        self.warmup = warmup
        self.obs_period = (self.warmup, self.max_simulation_time - self.warmup)
        self.k = len(arrival_rates)
        self.N = ciw.create_network(
            arrival_distributions={'Class ' + str(c): [dists.Exponential(arrival_rates[c])] for c in range(self.k)},
            service_distributions={'Class ' + str(c): [dists.Exponential(service_rates[c])] for c in range(self.k)},
            number_of_servers=[number_of_servers],
            class_change_time_distributions=[[dists.Exponential(rate) if rate is not None else None for rate in row] for row in class_change_rate_matrix],
            priority_classes=({'Class ' + str(c): c for c in range(self.k)}, [preempt])
        )
        self.run()
        self.aggregate_states()
        self.aggregate_states_by_class()
    
    def run(self):
        """
        Runs the simulation and finds the state probabilities
        """
        ciw.seed(0)
        self.Q = ciw.Simulation(self.N, tracker=trackers.NodeClassMatrix())
        self.Q.simulate_until_max_time(self.max_simulation_time, progress_bar=True)
        self.probs = self.Q.statetracker.state_probabilities(observation_period=self.obs_period)

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
