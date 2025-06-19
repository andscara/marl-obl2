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
    if node.visits == 0:
        return float("inf")

    agent_idx = node.game.agent_name_mapping[node.agent]
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits)

def uct(node: MCTSNode) -> MCTSNode:
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        simulations: int=100,
        rollouts: int=10,
        selection: Callable[[MCTSNode], MCTSNode]=uct,
        max_depth=None,
        eval_name: str = "eval"
    ) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
            max_depth: max depth for rollout
            eval_name: name of eval function to use
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        self.max_depth = max_depth
        self.eval_name = eval_name

    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> tuple[ActionType, float]:

        root = MCTSNode(parent=None, game=self.game.clone(), action=None)
        for i in range(self.simulations):
            # selection
            node = self.select_node(node=root)

            # expansion
            child = self.expand_node(node)

            # rollout
            rewards = self.rollout(child)

            #update values / Backprop
            self.backprop(child, rewards)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        # TODO
        # cumulate rewards and visits from node to root navigating backwards through parent
        while node is not None:
            node.cum_rewards += rewards
            node.visits += 1
            if node.parent is not None:
                node.value = node.cum_rewards[node.game.agent_name_mapping[node.parent.agent]] / node.visits
            node = node.parent

    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))
        # TODO
        # implement rollout policy

        # Si el juego está terminado, evitar rollouts.
        if node.game.terminated():
            for i, agent_id in enumerate(node.game.agents):
                rewards[i] += node.game.rewards[agent_id]

            return rewards

        for i in range(self.rollouts):
            sim  = node.game.clone()
            steps = 0
            eval_func = getattr(sim, self.eval_name, None)
            while not sim.terminated() and (self.max_depth <= 0 or steps < self.max_depth):
                action = np.random.choice(sim.available_actions())
                sim.step(action)
                steps += 1

            for i, agent_id in enumerate(sim.agents):
                if sim.terminated():
                    rewards[i] += sim.rewards[agent_id]
                else:
                    rewards[i] += eval_func(agent_id)

        return rewards / float(self.rollouts)

    def select_node(self, node: MCTSNode):
        while not node.game.terminated() and len(node.children) == len(node.game.available_actions()):
            node = self.selection(node)
        return node

    def expand_node(self, node) -> MCTSNode:
        # TODO
        # if the game is not terminated: 
        #    play an available action in node
        #    create a new child node and add it to node children
        if node.game.terminated():
            return node

        actions = node.game.available_actions()

        # Elimino acciones de hijos ya visitados
        # (acciones que ya han sido exploradas)
        for child in node.children:
            if child.action in actions:
                actions.remove(child.action)

        if len(actions) == 0:
            return node

        # Elijo una acción que lleve a un estado no visitado
        action = random.choice(actions)

        # Clonar juego y moverme al próximo estado
        child_game = node.game.clone()
        child_game.step(action)

        # Crear nodo hijo
        child_node = MCTSNode(parent=node, game=child_game, action=action)
        child_node.agent = child_game.agent_selection
        node.children.append(child_node)

        return child_node

    def action_selection(self, node: MCTSNode) -> tuple[ActionType, float]:
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
            action = np.random.choice(node.game.available_actions())
            value = 0.0
        return action, value
