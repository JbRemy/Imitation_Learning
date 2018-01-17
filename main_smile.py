from smile import SMILE

from parameters import parameters

simulator = SMILE(parameters)
simulator.run_episodes(agent=simulator.agents[0],
                       nb_episodes=1)
simulator.run_simulator(nb_episodes=5, nb_iterations=1)
