import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def random_dangerous_grid_world( env ):
	"""
	Performs a random trajectory on the given Dangerous Grid World environment 
	
	Args:
		environment: OpenAI Gym environment
		
	Returns:
		trajectory: an array containing the sequence of states visited by the agent
	"""
	trajectory = []
	current_pos = env.random_initial_state()
	print( f"Random trajectory generated starting from position {current_pos}:" )

	for _ in range(10):
		trajectory.append(current_pos)
		action = numpy.random.randint(0, env.action_space)
		current_pos = env.sample(action, current_pos)
		print( f"\tFrom position {trajectory[-1]} selected action {action} and moved to position {current_pos}" )

		if env.is_terminal(current_pos): 
			break
	
	return trajectory


class RecyclingRobot():
	"""
	Class that implements the environment Recycling Robot of the book: 'Reinforcement
	Learning: an introduction, Sutton & Barto'. Example 3.3 page 52 (second edition).
		
	Attributes
	----------
		observation_space : int
			define the number of possible actions of the environment
		action_space: int
			define the number of possible states of the environment
		actions: dict
			a dictionary that translate the 'action code' in human languages
		states: dict
			a dictionary that translate the 'state code' in human languages

	Methods
	-------
		reset( self )
			method that reset the environment to an initial state; returns the state
		step( self, action )
			method that perform the action given in input, computes the next state and the reward; returns 
			next_state and reward
		render( self )
			method that print the internal state of the environment
	"""


	def __init__( self ):

		# Loading the default parameters
		self.alfa = 0.7
		self.beta = 0.7
		self.r_search = 0.5
		self.r_wait = 0.2
		self.r_rescue = -3
		self.r_charge = 0

		# Defining the environment variables
		self.observation_space = 2
		self.action_space = 3
		self.actions = {0: "search", 1: "wait", 2: "recharge"}
		self.states = {0: "high", 1: "low"}


	def reset( self ):
		self.state = 0
		return self.state


	def step( self, action ):
		reward = 0

		if action == 0: # Search
			if self.state == 0: # High
				if numpy.random.rand() < self.alfa:
					self.state = 0
					reward = self.r_search
				else:
					self.state = 1
					reward = self.r_search

			else: # Low
				if numpy.random.rand() < 1 - self.beta:
					self.state = 0
					reward = self.r_rescue
				else:
					self.state = 1
					reward = self.r_search

		if action == 1: # Wait
			if self.state == 0: # High
				self.state = 0
				reward = self.r_wait

			else: # Low
				self.state = 1
				reward = self.r_wait

		if action == 2: # Recharge
			if self.state == 1:
				self.state = 0
				reward = self.r_charge

		return self.state, reward, False, None


	def render( self ):
		print( f"\nRecycling Robot Environment: \n\tStates: {self.states} \n\tActions: {self.actions}\n\n" )
		return True


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the second lesson of the RL-Lab!  *" )
	print( "*             (MDP and Environments)           *" )
	print( "************************************************" )

	print( "\nA) Random Policy on Dangerous Grid World:" )
	env = GridWorld()
	env.render()
	random_trajectory = random_dangerous_grid_world( env )
	print( "\nRandom trajectory generated:", random_trajectory )


	print( "\nB) Custom Environment: Recycling Robot" )
	env = RecyclingRobot()
	state = env.reset()
	env.render()
	ep_reward = 0
	
	for step in range(10):
		a = numpy.random.randint( 0, env.action_space )
		new_state, r, _, _ = env.step( a )
		ep_reward += r
		print( f"\tFrom state '{env.states[state]}' selected action '{env.actions[a]}': \t total reward: {ep_reward:1.1f}" )
		state = new_state


if __name__ == "__main__":
	main()
