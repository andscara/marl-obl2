import numpy as np
from numpy import ndarray
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, num_actions: int, obs: ObsType) -> None:
        self.obs = obs
        self.num_actions = num_actions
        self.cum_regrets = np.zeros(self.num_actions)
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

    def regret_matching(self):
        suma = np.sum([max(g, 0) for g in self.cum_regrets])
        if suma > 0:
            self.curr_policy = np.array([max(g, 0) for g in self.cum_regrets]) / suma
        else:
            self.curr_policy = np.full(self.num_actions, 1/self.num_actions)

        self.learned_policy = self.sum_policy / np.sum(self.sum_policy)

    def update(self, utility, node_utility, probability, agent_idx) -> None:
        # update
        product_p = np.prod([prob for q, prob in enumerate(probability) if q != agent_idx])
        self.cum_regrets += (utility - node_utility) * product_p
        self.sum_policy += probability[agent_idx] * self.curr_policy

        # regret matching policy
        self.regret_matching()

    def policy(self):
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            print('Node does not exist. Playing random.')
            return np.random.choice(self.game.available_actions())
    
    def train(self, niter=1000):
        for _ in range(niter):
            _ = self.cfr()

    def cfr(self):
        game = self.game.clone()
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability)

        return utility

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        # TODO
        if game.terminated():
            return game.reward(agent)

        agent_q = game.agent_selection
        agent_q_index = game.agent_name_mapping[agent_q]

        I = game.observe(agent_q)
        if type(I) == np.ndarray:
            I = tuple(I.reshape(-1))

        if I not in self.node_dict:
            # Se cambia el código base para no pasarle el juego,
            # al resetear el ambiente puede cambiar el jugador inicial
            # por lo que el nodo puede quedar con un juego clonado mal configurado
            # entonces en el update también se cambia para recibir
            # explicitamente el jugador que actualiza.
            self.node_dict[I] = Node(game.num_actions(agent_q), I)
        node = self.node_dict[I]

        actions = list(game.action_iter(agent_q))
        utility = np.zeros(len(actions))
        for action_index, action in enumerate(actions):
            g = game.clone()
            g.step(action)
            P = probability.copy()
            P[agent_q_index] *= node.curr_policy[action_index]
            utility[action_index] = self.cfr_rec(g, agent, P)

        node_utility = 0
        for a in range(node.num_actions):
            node_utility += node.curr_policy[a] * utility[a]

        if agent == agent_q:
            # se cambia el código base para pasar al nodo el agente que actualiza.
            node.update(utility, node_utility, probability, game.agent_name_mapping[agent])

        return node_utility

if __name__ == "__main__":
    from collections import OrderedDict
    from games.kuhn.kuhn import KuhnPoker

    game = KuhnPoker()
    agent_classes = [ CounterFactualRegret, CounterFactualRegret ]
    my_agents = {}
    game.reset()
    print(f"game initial player: {game.initial_player}")
    for i, agent in enumerate(game.agents):
        my_agents[agent] = agent_classes[i](game=game, agent=agent)

    for agent in game.agents:
        print('Training agent ' + agent)
        my_agents[agent].train(100000)
        print('Agent ' + agent + ' policies:')
        print(OrderedDict(map(lambda n: (n, my_agents[agent].node_dict[n].policy()), sorted(my_agents[agent].node_dict.keys()))))
        print('')