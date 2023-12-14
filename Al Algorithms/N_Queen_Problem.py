import math
import random


def attackOnStraightLine(board):
    attacks = 0
    queensInLine = 0

    for i in range(0, len(board)):
        for j in range(0, len(board)):
            if board[i][j] == 'Q':
                queensInLine += 1

        attacks += queensInLine if queensInLine > 1 else 0
        queensInLine = 0

    return math.ceil(attacks / 2)


def attackOnDiagonal(board):
    attacks = 0
    queensPos = []

    for i in range(0, len(board)):
        for j in range(0, len(board)):
            if board[j][i] == 'Q':
                queensPos.append((j, i))

    for pos1 in queensPos:
        for pos2 in queensPos:
            if pos1 != pos2 and abs(pos1[0] - pos2[0]) == abs(pos1[1] - pos2[1]):
                attacks += 1

    return int(attacks / 2)


def calculateHeuristic(board):
    return attackOnStraightLine(board) + attackOnDiagonal(board)


class NQueenProblem:
    def __init__(self, n):
        self.numberOfQueens = n
        self.board = [['-' for i in range(n)] for j in range(n)]
        self.boardConfig()
        self.minima = {
            'Hill Climbing': 0,
            'Simulated Annealing': 0
        }

    def boardConfig(self):
        # placing queens one column each
        for i in range(0, self.numberOfQueens):
            self.board[random.randint(0, self.numberOfQueens - 1)][i] = 'Q'

    def printBoard(self):
        print("Board : ")
        for i in range(0, self.numberOfQueens):
            for j in range(0, self.numberOfQueens):
                print(self.board[i][j], end=" ")
            print()
        print("Queens Attacking = ", calculateHeuristic(self.board))

    def hillClimbing(self):
        print("\nHill Climbing : ")
        self.printBoard()
        moves, bestHeuristic, tempBoard = 0, calculateHeuristic(self.board), self.board.copy()

        for i in range(0, self.numberOfQueens):
            for j in range(0, self.numberOfQueens):
                if tempBoard[i][j] == 'Q':
                    for k in range(0, self.numberOfQueens):
                        if i != k:
                            tempBoard[k][j] = 'Q'
                            tempBoard[i][j] = '-'

                            tempHeuristic = calculateHeuristic(tempBoard)

                            if tempHeuristic == 0:
                                print("Solution Found!\nMoves = ", moves)
                                self.printBoard()
                                return

                            if tempHeuristic < bestHeuristic:
                                bestHeuristic = tempHeuristic
                                self.board = tempBoard.copy()
                                i, j = -1, -1
                                moves += 1
                                self.printBoard()

                            else:
                                if moves == 0:
                                    print("No Solution Found!\nMoves Taken = ", moves)
                                else:
                                    print("Best Possible Solution Found.\nMoves = ", moves)
                                self.minima['Hill Climbing'] = bestHeuristic
                                return

    def simulatedAnnealing(self):
        print("\nSimulated Annealing : ")
        self.printBoard()
        temp, moves = 1.0, 0
        bestHeuristic, tempBoard = calculateHeuristic(self.board), self.board.copy()

        while temp > 0.001:
            i = random.randint(0, self.numberOfQueens - 1)
            j = random.randint(0, self.numberOfQueens - 1)

            if tempBoard[i][j] == 'Q':
                k = random.randint(0, self.numberOfQueens - 1)
                if k != i:
                    tempBoard[k][j] = 'Q'
                    tempBoard[i][j] = '-'

                tempHeuristic = calculateHeuristic(tempBoard)
                if tempHeuristic == 0:
                    print("Solution Found!\nMoves = ", moves)
                    self.printBoard()
                    return

                deltaE = bestHeuristic - tempHeuristic

                if deltaE > 0:
                    bestHeuristic = tempHeuristic
                    moves += 1
                    self.board = tempBoard.copy()

                else:
                    probability = math.exp(deltaE / temp)
                    if probability > random.uniform(0, 1):
                        bestHeuristic = tempHeuristic
                        moves += 1
                        self.board = tempBoard.copy()
                    else:
                        tempBoard = self.board.copy()
                temp *= 0.95

        self.minima['Simulated Annealing'] = bestHeuristic

        if moves == 0:
            print("No Solution Found!\nMoves Taken = ", moves)
        else:
            print("Best Possible Solution Found.\nMoves = ", moves)
            self.printBoard()


if __name__ == '__main__':
    queens = 0
    while not (5 <= queens <= 10):
        queens = int(input("Enter The Number of Queens (5-10) : "))

    problem = NQueenProblem(queens)
    Board = problem.board.copy()

    problem.hillClimbing()

    problem.board = None
    problem.board = Board.copy()
    problem.simulatedAnnealing()

    print("Minima (The Lesser The Better) :\n", problem.minima)
