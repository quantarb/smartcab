import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np 

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q = {}
        #self.epsilon = 1 # completely random
        # self.epsilon = 0 # always picks the move with the highest q value
        self.epsilon = 0.10
        self.state = ""
        self.alpha = 0.20
        self.successful_trips = 0
        self.total_moves = 0
        self.total_rewards = 0
        self.total_violations = 0 
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)

        
        print "Total number of moves: {}".format(self.total_moves)
        print "Total rewards: {}".format(self.total_rewards)
        print "Total number of violations: {}".format(self.total_violations)
        print "Total number of states explored: {}".format(len(self.q.keys()))
        print "Total number of successful trips: {}".format(self.successful_trips)        
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        q = self.q           
        state = self.state 
        list_of_actions=["forward","right","left", None] 
        number_of_actions = len(list_of_actions)
        
        # TODO: Update state
        state = "next_waypoint: {}, {}".format(self.next_waypoint, str(inputs)).replace('{', '').replace('}', '')
        self.state = state
        
        # TODO: Select action according to your policy     
        if state not in q:
            q[state] = np.zeros(number_of_actions)
                                    
        index = random.choice(list(enumerate(list_of_actions)))[0] if random.random() < self.epsilon else np.argmax(q[state])
        
        action = list_of_actions[index]; 
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_rewards += reward
        if reward > 10:
            self.successful_trips += 1 
        elif reward < 0:
            self.total_violations += 1
        self.total_moves += 1
        # TODO: Learn policy based on state, action, reward
        
        q[state][index] = (1 - self.alpha) * q[state][index] + self.alpha * reward         
        
        
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
