from base.agent import Agent, AgentID
from base.game import AlternatingGame
import numpy as np
import sys

class MiniMax(Agent):

    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        seed=None,
        depth: int=sys.maxsize,
        eval_name: str = "eval"
    ) -> None:
        super().__init__(game, agent)

        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        self.depth = depth
        self.seed = seed
        self.eval_name = eval_name
        np.random.seed(seed)
    
    def action(self):
        act, _ = self.minimax(self.game, self.depth)
        return act

    def minimax(self, game: AlternatingGame, depth: int):

        agent = game.agent_selection
        chosen_action = None  

        #Casos base

        if game.terminated():             
            return None, game.reward(self.agent)

        if depth == 0:
            return None, self.eval(game)
        
        #Casos no base

        actions = game.available_actions()
        np.random.shuffle(actions)
        action_nodes = []
        for action in actions:
            child = game.clone()
            child.step(action)
            action_nodes.append((action, child))

        if agent != self.agent: # Min
            value = float('inf')
            for action, child in action_nodes:
                _, minimax_value = self.minimax(child, depth-1)
                if minimax_value < value:
                    value = minimax_value
                    chosen_action = action

        else: # Max (player == self.player)
            value = float('-inf')
            for action, child in action_nodes:
                _, minimax_value = self.minimax(child, depth-1)
                if minimax_value > value:
                    value = minimax_value
                    chosen_action = action

        return chosen_action, value

    def eval(self, game: AlternatingGame):
        eval_func = getattr(game, self.eval_name, None)
        return eval_func(self.agent)
