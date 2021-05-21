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

        "*** YOUR CODE HERE ***"

        for ghostState in newGhostStates:
            # If distance of pacman position and ghost position less than 1 do not do action
            if util.manhattanDistance(newPos, ghostState.getPosition()) <= 1:
                return -1

        if newPos in currentGameState.getFood().asList():
            isFood = 1.0
        else:
            isFood = 0.0

        newFoodList = newFood.asList()
        manhattanFoodDistance = [util.manhattanDistance(newPos, foodPos) for foodPos in newFoodList]

        closestFoodDist = min(manhattanFoodDistance, default=1)

        return float(1.0 / closestFoodDist) + isFood


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


        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(agent, depth, gameState):

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            elif agent == 0:
                return max_value(agent, depth, gameState)

            elif agent > 0:
                return min_value(agent, depth, gameState)

        def min_value(agent, depth, gameState):
            agentnext = agent + 1

            if gameState.getNumAgents() == agentnext:
                agentnext = 0
                depth += 1

            children = [gameState.generateSuccessor(agent, direction) for direction in gameState.getLegalActions(agent)]
            miniMaxOfChildren = [minimax(agentnext, depth, child) for child in children]
            minVal = min(miniMaxOfChildren)

            return minVal

        def max_value(agent, depth, gameState):
            children = [gameState.generateSuccessor(agent, direction) for direction in gameState.getLegalActions(agent)]
            miniMaxOfChildren = [minimax(1, depth, child) for child in children]
            maxVal = max(miniMaxOfChildren)
            return maxVal

        pacmanDirections = gameState.getLegalActions(0)
        miniMaxPacmanDirections = [minimax(1, 0, gameState.generateSuccessor(0, step)) for step in pacmanDirections]
        total = max(miniMaxPacmanDirections)
        indexOfMax = miniMaxPacmanDirections.index(total)
        direction = pacmanDirections[indexOfMax]

        return direction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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

        def expectimax(agent, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            elif agent == 0:
                return max_value(agent, depth, gameState)

            else:
                return exp_value(agent, depth, gameState)

        def max_value(agent, depth, gameState):
            children = [gameState.generateSuccessor(agent, direction) for direction in gameState.getLegalActions(agent)]
            miniMaxOfChildren = [expectimax(1, depth, child) for child in children]
            maxVal = max(miniMaxOfChildren)
            return maxVal

        def exp_value(agent, depth, gameState):
            agentNext = agent + 1

            if gameState.getNumAgents() == agentNext:
                agentNext = 0
                depth += 1

            successorsOfDirections = [gameState.generateSuccessor(agent, dir) for dir in
                                      gameState.getLegalActions(agent)]
            expectiMaxOfDirections = [expectimax(agentNext, depth, successor) for successor in successorsOfDirections]
            sumOfExpectimax = sum(expectiMaxOfDirections)

            average = sumOfExpectimax / len(gameState.getLegalActions(agent))

            return average

        pacmanDirections = gameState.getLegalActions(0)
        miniMaxPacmanDirections = [expectimax(1, 0, gameState.generateSuccessor(0, step)) for step in pacmanDirections]
        total = max(miniMaxPacmanDirections)
        indexOfMax = miniMaxPacmanDirections.index(total)
        direction = pacmanDirections[indexOfMax]

        return direction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()

    distanceFoods = [util.manhattanDistance(newPos, food) for food in newFood]
    minDistanceFood = min(distanceFoods, default=1)

    ghostDistances = [util.manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates]

    for i in range(len(ghostDistances)):
        if ghostDistances[i] == 0 and newGhostStates[i].scaredTimer == 0:
            return -float('inf')
        if ghostDistances[i] == 1 and newGhostStates[i].scaredTimer == 0:
            minDistanceFood -= 1000
        if ghostDistances[i] < newGhostStates[i].scaredTimer:
            minDistanceFood += 500

    return currentGameState.getScore() + 1.0 / minDistanceFood


# Abbreviation
better = betterEvaluationFunction
