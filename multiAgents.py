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
from math import inf


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
        bestIndices = [index for index in range(len(scores)) if
                       scores[index] == bestScore]  # There may be more than one action that gives the same score
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        #print('SUCCESSOR GAME STATE {}'.format(successorGameState))
        #print('NEW POSITION {}'.format(newPos))
        #print('NEW FOOD: {}'.format(newFood))
        #print('NEW GHOSTS STATES: {}'.format(newGhostStates[0]))
        #print('NEW SCARED TIMES {}'.format(newScaredTimes[0]))

        "*** YOUR CODE HERE ***"
        # Handling the case of a successor leading to winning or losing
        if successorGameState.isWin():
            return float("inf")
        elif successorGameState.isLose():
            return -float("inf")

        eval_score = 0

        # Get the distance between pacman and the ghosts
        pacman_ghosts_distances = map(lambda x: manhattanDistance(x.configuration.getPosition(), newPos),
                                      newGhostStates)
        # Get the distance between pacman and food
        pacman_food_distances = map(lambda x: manhattanDistance(x, newPos), newFood.asList())

        # Adding a penalty if pacman gets near a ghost, or away from food
        eval_score -= 70 * 1 / min(pacman_ghosts_distances)
        eval_score += 10 * 1 / min(pacman_food_distances)

        return successorGameState.getScore() + eval_score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        NumAgents = gameState.getNumAgents()

        def max_value(gameState, current_depth):
            if gameState.isWin() or gameState.isLose() or (current_depth == self.depth):
                return self.evaluationFunction(gameState)

            legalMovesPacman = gameState.getLegalActions(0)
            successorGameStatePacman = [gameState.generateSuccessor(0, action) for action in legalMovesPacman]

            maxi = -inf

            for PacmanState in successorGameStatePacman:
                maxi = max(maxi, min_value(PacmanState, current_depth, 1))

            return maxi

        def min_value(gameState, current_depth, current_Agent):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            mini = +inf

            for action in gameState.getLegalActions(current_Agent):

                if current_Agent == NumAgents - 1:  # The last ghost
                    mini = min(mini, max_value(gameState.generateSuccessor(current_Agent, action), current_depth + 1))

                else:  # There are still more ghosts
                    mini = min(mini, min_value(gameState.generateSuccessor(current_Agent, action), current_depth,
                                               current_Agent + 1))

            return mini

        return max(((min_value(gameState.generateSuccessor(0, action), 0, 1), action) for action in
                    gameState.getLegalActions(0)), key=lambda entry: entry[0])[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        NumAgents = gameState.getNumAgents()

        def max_value(gameState, current_depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or (current_depth == self.depth):
                return self.evaluationFunction(gameState), None

            legalMovesPacman = gameState.getLegalActions(0)

            maxi = -inf
            best_action = None
            for action in legalMovesPacman:
                current_utility = min_value(gameState.generateSuccessor(0, action), current_depth, 1, alpha, beta)
                if current_utility > maxi:
                    maxi = current_utility
                    best_action = action

                if maxi > beta:
                    return maxi, best_action
                alpha = max(alpha, maxi)

            return maxi, best_action

        def min_value(gameState, current_depth, current_Agent, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalMovesGhost = gameState.getLegalActions(current_Agent)

            mini = +inf
            for action in legalMovesGhost:

                if current_Agent == NumAgents - 1:  # The last ghost
                    mini = min(mini, max_value(gameState.generateSuccessor(current_Agent, action), current_depth + 1, alpha, beta)[0])
                    if mini < alpha:
                        return mini
                    beta = min(beta, mini)

                else:  # There are still more ghosts
                    mini = min(mini, min_value(gameState.generateSuccessor(current_Agent, action), current_depth,
                                               current_Agent + 1, alpha, beta))
                    if mini < alpha:
                        return mini
                    beta = min(beta, mini)

            return mini

        return max_value(gameState, 0, -inf, +inf)[1]


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

        def max_value(gameState, current_depth):
            if gameState.isWin() or gameState.isLose() or (current_depth == self.depth):
                return self.evaluationFunction(gameState)

            legalMovesPacman = gameState.getLegalActions(0)
            successorGameStatePacman = [gameState.generateSuccessor(0, action) for action in legalMovesPacman]

            maxi = -inf
            for PacmanState in successorGameStatePacman:
                maxi = max(maxi, chance_node(PacmanState, current_depth, 1))
            return maxi

        # terminal_test: isWin() / isLose()
        # utility: score ----->player
        # actions: LegalActions()
        # result: generateSuccessor()

        def chance_node(gameState, current_depth, current_Agent):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            chance = 0
            numAgents = gameState.getNumAgents()
            legalMovesGhost = gameState.getLegalActions(current_Agent)

            prob = 1 / len(legalMovesGhost)
            for action in gameState.getLegalActions(current_Agent):
                if current_Agent == numAgents - 1:  # The last ghost

                    # Get the expected value of the utilities
                    chance += prob * max_value(gameState.generateSuccessor(current_Agent, action), current_depth + 1)

                else:  # There are still more ghosts

                    # Get the expected value of the utilities
                    chance += prob * chance_node(gameState.generateSuccessor(current_Agent, action), current_depth,
                                                 current_Agent + 1)
            return chance

        return max(((chance_node(gameState.generateSuccessor(0, action), 0, 1), action) for action in
                    gameState.getLegalActions(0)), key=lambda entry: entry[0])[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    eval_score = 0

    newWalls = currentGameState.getWalls()

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

    # Handling the case of a successor leading to winning or losing
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")

    # Get the distance between pacman and the ghosts
    pacman_ghosts_distances = map(lambda x: manhattanDistance(x.configuration.getPosition(), newPos),
                                  newGhostStates)
    # Get the distance between pacman and food

    pacman_food_distances = map(lambda x: manhattanDistance(x, newPos), newFood.asList())

    # Get the distance between pacman and capsules
    pacman_capsules_distances = map(lambda x: manhattanDistance(x, newPos), newCapsules)

    # Adding a penalty if pacman gets near a ghost, or away from food

    if newScaredTimes == 0:  # Get away from ghosts only if they're not scared
        eval_score -= 40 * 1 / min(pacman_ghosts_distances)
    else:
        eval_score += 40 * 1 / min(pacman_ghosts_distances)

    eval_score += 10 * 1 / min(pacman_food_distances)

    if len(newCapsules) != 0:
        eval_score += 10 * 1 / min(pacman_capsules_distances)

    return currentGameState.getScore() + eval_score


# Abbreviation
better = betterEvaluationFunction
