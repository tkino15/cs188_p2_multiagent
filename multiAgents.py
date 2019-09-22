# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        def get_manhattan_distance(loc1, loc2):
            return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

        # current food
        currentFood = currentGameState.getFood()
        currentCaps = currentGameState.getCapsules()
        foods = currentFood.asList() + currentCaps

        # min distance bw pacman and foods
        food_distance = float('inf')
        for food in foods:
            food_distance = min(get_manhattan_distance(newPos, food), food_distance)

        # min distance bw pacman and ghosts
        ghost_distance = float('inf')
        for ghost in newGhostStates:
            ghost_distance = min(get_manhattan_distance(newPos, ghost.getPosition()), ghost_distance)

        evaluation = \
            -1 * food_distance + \
            -1 * 1 / (ghost_distance + 1e-10)  # risk of ghost become extremely large when it is very closed.

        # print(action, food_distance, 1/ghost_distance, evaluation)

        return evaluation

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(current_state, current_depth, current_index, max_depth):

            next_actions = current_state.getLegalActions(current_index)

            # base case
            if current_depth >= max_depth or not next_actions or current_state.isWin() or current_state.isLose():
                return ('', self.evaluationFunction(current_state))

            if current_index % n_agents == 0:  # maximizer turn
                max_score = -float('inf')
                max_action = ''
                for next_action in next_actions:
                    next_score = minimax(current_state.generateSuccessor(current_index, next_action),
                                         current_depth + 1, ((current_index + 1) % n_agents), max_depth)[1]
                    if max_score < next_score:
                        max_score = next_score
                        max_action = next_action

                return max_action, max_score

            else:  # minimizer turn
                min_score = float('inf')
                min_action = ''
                for next_action in next_actions:
                    next_score = minimax(current_state.generateSuccessor(current_index, next_action),
                                         current_depth + 1, ((current_index + 1) % n_agents), max_depth)[1]
                    if min_score > next_score:
                        min_score = next_score
                        min_action = next_action

                return min_action, min_score

        # extract conditions
        n_agents = gameState.getNumAgents()
        max_depth = self.depth * n_agents + 1

        return minimax(gameState, 1, 0, max_depth)[0]  # return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(current_state, current_depth, current_index, max_depth, alpha, beta):

            next_actions = current_state.getLegalActions(current_index)

            # base case
            if current_depth >= max_depth or not next_actions or current_state.isWin() or current_state.isLose():
                return ('', self.evaluationFunction(current_state))

            if current_index % n_agents == 0:  # maximizer turn
                max_score = -float('inf')
                max_action = ''
                for next_action in next_actions:
                    next_score = alphabeta(current_state.generateSuccessor(current_index, next_action),
                                           current_depth + 1, ((current_index + 1) % n_agents), max_depth,
                                           alpha, beta)[1]
                    if max_score < next_score:
                        max_score = next_score
                        max_action = next_action

                    # alpha beta pruning
                    if max_score > beta:
                        return max_action, max_score

                    alpha = max(alpha, max_score)

                return max_action, max_score

            else:  # minimizer turn
                min_score = float('inf')
                min_action = ''
                for next_action in next_actions:
                    next_score = alphabeta(current_state.generateSuccessor(current_index, next_action),
                                           current_depth + 1, ((current_index + 1) % n_agents), max_depth,
                                           alpha, beta)[1]
                    if min_score > next_score:
                        min_score = next_score
                        min_action = next_action

                    # alpha beta pruning
                    if min_score < alpha:
                        return min_action, min_score

                    beta = min(beta, min_score)

                return min_action, min_score

        # extract conditions
        n_agents = gameState.getNumAgents()
        max_depth = self.depth * n_agents + 1

        # initialize alpha beta
        alpha, beta = -float('inf'), float('inf')

        return alphabeta(gameState, 1, 0, max_depth, alpha, beta)[0]  # return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(current_state, current_depth, current_index, max_depth):

            next_actions = current_state.getLegalActions(current_index)

            # base case
            if current_depth >= max_depth or not next_actions or current_state.isWin() or current_state.isLose():
                return ('', self.evaluationFunction(current_state))

            if current_index % n_agents == 0:  # maximizer turn
                max_score = -float('inf')
                max_action = ''
                for next_action in next_actions:
                    next_score = expectimax(current_state.generateSuccessor(current_index, next_action),
                                         current_depth + 1, ((current_index + 1) % n_agents), max_depth)[1]
                    if max_score < next_score:
                        max_score = next_score
                        max_action = next_action

                return max_action, max_score

            else:  # minimizer turn
                expect_score = 0
                for next_action in next_actions:
                    next_score = expectimax(current_state.generateSuccessor(current_index, next_action),
                                         current_depth + 1, ((current_index + 1) % n_agents), max_depth)[1]
                    expect_score += next_score/len(next_actions)

                return ('', expect_score)

        # extract conditions
        n_agents = gameState.getNumAgents()
        max_depth = self.depth * n_agents + 1

        return expectimax(gameState, 1, 0, max_depth)[0]  # return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    def get_manhattan_distance(loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    # if currentGameState.isWin():
    #     return float('inf')
    # if currentGameState.isLose():
    #     return -float('inf')

    pacman = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList() + currentGameState.getCapsules()
    walls = currentGameState.getWalls()

    # initialize queue
    closed = set()
    fringe = util.Queue()
    depth = 0
    fringe.push((pacman, depth))

    # BFS to nearest food
    while True:
        if fringe.isEmpty():
            break

        node, depth = fringe.pop()

        # break when search reaches any food
        if node in foods:
            break

        # check if node is closed list
        if node not in closed:
            closed.add(node)

            # expand
            for dx, dy in zip([1, -1, 0, 0], [0, 0, 1, -1]):
                x = node[0] + dx
                y = node[1] + dy
                if not walls[x][y]:
                    fringe.push(((x, y), depth + 1))

    n_food_left = len(foods)

    eval = currentGameState.getScore() + -1 * depth + -10 * n_food_left

    return eval

# Abbreviation
better = betterEvaluationFunction
