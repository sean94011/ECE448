# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    from collections import deque

    queue = deque([])
    visited = []
    path = []

    path.append(maze.getStart())
    queue.append(path)

    while queue:
        cur_path = queue.popleft()
        cur_pos = cur_path[len(cur_path)-1]
    
        cur_row, cur_col = cur_pos

        visited.append(cur_pos)

        if cur_pos in maze.getObjectives():
            print(cur_path)
            return cur_path

        else:
            for next_pos in maze.getNeighbors(cur_row,cur_col):
                if next_pos in visited:
                    continue
                temp_path = cur_path + [next_pos]
                queue.append(temp_path)

    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    from queue import PriorityQueue

    start_pos = maze.getStart()

    frontier = PriorityQueue()
    frontier.put(start_pos,0)

    
    predecessor = dict()
    predecessor[start_pos] = None

    cost = dict()
    cost[start_pos] = 0
    
    goals = maze.getObjectives()
    goal_pos = goals[0]
    goal_row, goal_col = goal_pos

    while not frontier.empty():
        cur_pos = frontier.get()
        cur_row, cur_col = cur_pos

        if cur_pos in maze.getObjectives():
            break

        for next_pos in maze.getNeighbors(cur_row,cur_col):
            next_row, next_col = next_pos
            if not maze.isValidMove(next_row, next_col):
                continue
            temp_cost = cost[cur_pos] + abs(cur_row-next_row) + abs(cur_col-next_col)
            if next_pos not in cost or temp_cost < cost[next_pos]:
                cost[next_pos] = temp_cost
                f = temp_cost + abs(goal_row-next_row) + abs(goal_col-next_col)
                frontier.put(next_pos,f)
                predecessor[next_pos] = cur_pos
                print(predecessor)

    path = []
    while cur_pos != None:
        path.append(cur_pos)
        cur_pos = predecessor[cur_pos]
    path.reverse
    return path

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
