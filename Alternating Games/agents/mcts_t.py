from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import random
import numpy as np
from typing import Callable

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent               # Nodo padre (None si es la raíz)
        self.game = game                   # Estado del juego en este nodo
        self.action = action               # Acción que llevó a este estado desde el padre
        self.children = []                 # Lista de nodos hijos
        self.explored_children = 0         # Contador de hijos ya “visitados” en selección
        self.visits = 0                    # Cantidad de simulaciones que pasaron por aquí
        self.value = 0                     # Valor promedio de recompensa para el agente que movió
        self.cum_rewards = np.zeros(len(game.agents))  # Suma de recompensas por agente
        self.agent = self.game.agent_selection         # Agente que moverá en este estado

def ucb(node, C=sqrt(2)) -> float:
    agent_idx = node.game.agent_name_mapping[node.agent]
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits)

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> (ActionType, float):

        root = MCTSNode(parent=None, game=self.game, action=None)

        for i in range(self.simulations):

            node = root
            node.game = self.game.clone()

            #print(i)
            #node.game.render()

            # selection
            #print('selection')
            node = self.select_node(node=node)

            # expansion
            #print('expansion')
            self.expand_node(node)

            # rollout
            #print('rollout')
            rewards = self.rollout(node)

            #update values / Backprop
            #print('backprop')
            self.backprop(node, rewards)

        #print('root childs')
        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        # TODO
        # cumulate rewards and visits from node to root navigating backwards through parent
        while node is not None:
            agent_idx = node.game.agent_name_mapping[node.agent]
            node.cum_rewards[agent_idx] += rewards[agent_idx]
            node.visits += 1
            node.value = node.cum_rewards[agent_idx] / node.visits
            node = node.parent
        pass

    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))
        # TODO
        # implement rollout policy
        # for i in range(self.rollouts): 
        #     play random game and record average rewards
        for i in range(self.rollouts):
            sim  = node.game.clone()
            while not sim .terminated:
                action = sim .random_action(node.agent)
                sim .step(action)
            for i, agent_id in enumerate(sim.agents):
                rewards[i] += sim.rewards[agent_id]
        return rewards / float(self.rollouts)

    def select_node(self, node: MCTSNode):
        curr_node = node

        while curr_node.children:
            if curr_node.explored_children < len(curr_node.children):
                idx = curr_node.explored_children
                curr_node.explored_children += 1
                curr_node = curr_node.children[idx]
            else:
                best_child = None
                best_value = float("-inf")
                for hijo in curr_node.children:
                    val = ucb(hijo)
                    if val > best_value:
                        best_value = val
                        best_child = hijo
                curr_node = best_child

        return curr_node

    def expand_node(self, node) -> None:
        # TODO
        # if the game is not terminated: 
        #    play an available action in node
        #    create a new child node and add it to node children
        if not node.game.terminated():

            # Obtengo el agente actual y las acciones disponibles
            actions = node.game.available_actions()

            # Elimino acciones de hijos ya visitados
            # (acciones que ya han sido exploradas)
            for child in node.children:
                if child.action in actions:
                    actions.remove(child.action)

            if len(actions) == 0:
                return

            # Elijo una acción que lleve a un estado no visitado
            action = random.choice(actions)

            # Clonar juego y moverme al próximo estado	
            child_game = node.game.clone()
            child_game.step(action)

            # Crear nodo hijo
            child_node = MCTSNode(parent=node, game=child_game, action=action)
            child_node.agent = child_game.agent_selection
            node.children.append(child_node)

    def action_selection(self, node: MCTSNode) -> (ActionType, float):
        action: ActionType = None
        value: float = 0
        # TODO
        # hint: return action of child with max value 
        # other alternatives could be considered
        if node.children:
            best_child = max(node.children, key=lambda x: x.value)
            action = best_child.action
            value = best_child.value
        else:
            action = node.game.random_action(node.agent)
            value = 0.0
        return action, value    