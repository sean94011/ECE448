# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    queue =[]
    visited = []
    path = []
    
    start_pos = maze.getStart()
    goal_pos = maze.getObjectives()
    
    path.append(start_pos)
    queue.append(path)

    if start_pos in goal_pos:
        return path
    
    while queue:
        cur_path = queue.pop(0)
        cur_pos = cur_path[len(cur_path)-1]
        cur_row, cur_col = cur_pos

        if cur_pos in visited:
            continue
        else:
            for next_pos in maze.getNeighbors(cur_row,cur_col):
                temp_path = cur_path + [next_pos]
                if next_pos in goal_pos:
                    return temp_path
                queue.append(temp_path)
            visited.append(cur_pos)
            
    return None
