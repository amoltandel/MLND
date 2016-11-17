import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DIRECTION_DICT = {(1, 0):'East', (0, -1):'North', (-1, 0):'West', (0, 1):'South'}
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    cumulative_rewards = []
    success_count = 0
    invalid_move_count = []
    incorrect_move_count = []
    correct_move_count = []
    step_counts = []
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.location = None
        self.Q = defaultdict(dict)
        self.alpha = 0.4
        self.gamma = 0.1
        self.epsilon = 0.0
        self.input = None
        self.state = None
        self.next_state = None
        self.trials = 0
        self.cumulative_reward = 0.0
        self.correct_moves = 0.0
        self.invalid_moves = 0.0
        self.incorrect_moves = 0.0
        self.steps = 0.0
        LearningAgent.cumulative_rewards = []
        LearningAgent.success_count = 0
        LearningAgent.invalid_move_count = []
        LearningAgent.incorrect_move_count = []
        LearningAgent.correct_move_count = []
        LearningAgent.step_counts = []



    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trials += 1
        LearningAgent.cumulative_rewards.append(self.cumulative_reward)
        LearningAgent.correct_move_count.append(self.correct_moves)
        LearningAgent.incorrect_move_count.append(self.incorrect_moves)
        LearningAgent.invalid_move_count.append(self.invalid_moves)
        LearningAgent.step_counts.append(self.steps)
        self.cumulative_reward = 0.0
        self.correct_moves = 0.0
        self.invalid_moves = 0.0
        self.incorrect_moves = 0.0
        self.steps = 0.0


    def update(self, t):
        # Gather inputs
        #for decaying epsilon
        if self.trials == 0:
            self.epsilon = 1.0
        else:
            self.epsilon = 1/float(self.trials)
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = OrderedDict(self.env.sense(self))
        deadline = self.env.get_deadline(self)
        if inputs['light'] == 'green':
            self.steps += 1.0

        # TODO: Update state
        inputs['next_waypoint'] = self.next_waypoint
        #print inputs
        self.state = tuple(inputs.values())

        if self.state not in self.Q.keys():
            self.Q[self.state] = OrderedDict()
            self.Q[self.state]['forward'] = 0.0
            self.Q[self.state]['left'] = 0.0
            self.Q[self.state]['right'] = 0.0
            self.Q[self.state][None] = 0.0

        # TODO: Select action according to your policy
        # Execute action and get reward

        if random.random() < self.epsilon:
            action = random.choice(Environment.valid_actions)
        else:
            action = self.Q[self.state].keys()[np.argmax(self.Q[self.state].values())]

        #print "Action: {}".format(action)
        reward = self.env.act(self, action)
        self.cumulative_reward += reward
        if reward == -0.5:
            self.incorrect_moves += 1.0
        elif reward == -1.0:
            self.invalid_moves += 1.0
        elif reward == 12.0:
            LearningAgent.success_count += 1
        else:
            self.correct_moves += 1.0
        next_inp = OrderedDict(self.env.sense(self))
        next_inp['next_waypoint'] = self.planner.next_waypoint()

        self.next_state = tuple(next_inp.values())

        #print "Current State: {}\nNext State: {}".format(inputs, next_inp)

        if self.next_state not in self.Q.keys():
            self.Q[self.next_state] = OrderedDict()
            self.Q[self.next_state]['forward'] = 0.0
            self.Q[self.next_state]['left'] = 0.0
            self.Q[self.next_state]['right'] = 0.0
            self.Q[self.next_state][None] = 0.0

        # TODO: Learn policy based on state, action, reward
        self.Q[self.state][action] = (1.0 - self.alpha) * self.Q[self.state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[self.next_state].values()))
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it

    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    sim.run(n_trials=1000)  # run for a specified number of trials
    print "Successes:{}".format(LearningAgent.success_count)
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print LearningAgent.cumulative_rewards
    print LearningAgent.incorrect_move_count
    print LearningAgent.invalid_move_count
    print LearningAgent.correct_move_count
    print LearningAgent.step_counts

    '''
    To generate the heatmap, uncomment the following code block and comment the previous one

    '''
    # alphas = np.arange(0.0, 1.05, 0.05)
    # gammas = np.arange(0.0, 1.05, 0.05)
    # print alphas
    # #exit(1)
    #
    # heatmap = []
    # argmax_gamma = 0
    # argmax_alpha = 0
    # max_performance = 0.0
    # sim_data = []
    # for i, alpha in enumerate(alphas):
    #     row = []
    #     for j, gamma in enumerate(gammas):
    #         print "for alpha, gamma = {}, {}".format(alpha, gamma)
    #         e = Environment()
    #         a = e.create_agent(LearningAgent)
    #         a.alpha = alpha
    #         a.gamma = gamma
    #
    #         e.set_primary_agent(a, enforce_deadline=True)
    #         sim = Simulator(e, update_delay=0.0, display=False)
    #         sim.run(n_trials=100)
    #         print "Successful journeys : {}".format(LearningAgent.success_count)
    #
    #
    #         avg_cumulative_rewards = np.average(LearningAgent.cumulative_rewards)
    #         avg_invalid_moves = np.average(LearningAgent.invalid_move_count)
    #         avg_incorrect_moves = np.average(LearningAgent.incorrect_move_count)
    #         avg_correct_moves = np.average(LearningAgent.correct_move_count)
    #         avg_steps = np.average(LearningAgent.step_counts)
    #         #magnitude-wise, abs(correct move reward):abs(incorrect move reward):abs(invalid move reward) = 4:1:2
    #         performance = ((avg_correct_moves * 4 / (avg_incorrect_moves + avg_invalid_moves * 2)) + avg_cumulative_rewards) * LearningAgent.success_count/(100.0 * avg_steps)
    #         print "avg_cumulative_rewards:{}, avg_correct_moves:{}, avg_incorrect_moves:{}, avg_invalid_moves:{}, avg_steps:{}".format(avg_cumulative_rewards,avg_correct_moves, avg_incorrect_moves, avg_invalid_moves, avg_steps)
    #         sim_data.append(tuple([alpha, gamma, LearningAgent.success_count, avg_cumulative_rewards, avg_correct_moves, avg_incorrect_moves, avg_invalid_moves, avg_steps, performance]))
    #         print "Performance: {}".format(performance)
    #
    #         row.append(performance)
    #         if performance > max_performance:
    #             max_performance = performance
    #             argmax_gamma = j
    #             argmax_alpha = i
    #     heatmap.append(row)
    # print heatmap
    # print alphas[argmax_alpha]
    # print gammas[argmax_gamma]
    # np.savetxt('simdata.csv', sim_data, delimiter=',')
    # ax = sns.heatmap(heatmap, xticklabels=gammas, yticklabels=alphas, annot=True)
    # ax.set(xlabel="gamma", ylabel="alpha")
    # plt.show()
if __name__ == '__main__':
    run()
