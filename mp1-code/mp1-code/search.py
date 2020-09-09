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
import heapq as hq

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def m_dist(pos1_row, pos1_col, pos2_row, pos2_col):
    return abs(pos1_row-pos2_row) + abs(pos1_col-pos2_col)

def actual_dist(pos1_row, pos1_col, pos2_row, pos2_col):
    return abs(pos1_row-pos2_row)**2 + abs(pos1_col-pos2_col)**2

def mst_cost(start_node, objectives):
    dist = dict()
    predecessor = dict()
    visited = []

    graph = objectives

    for vertex in graph:
        dist[vertex] = float("inf")
        predecessor[vertex] = None

    dist[start_node] = 0
    
    pqueue = []
    hq.heapify(pqueue)
    hq.heappush(pqueue, (dist[start_node],start_node))

    for i in range(len(graph)):
        if not pqueue:
            break
        temp = hq.heappop(pqueue)
        cur_dist, cur_node = temp

        visited.append(cur_node)

        for v in graph:
            if v in visited:
                continue

            v_row, v_col = v
            cur_row, cur_col = cur_node
            temp_cost = m_dist(v_row, v_col, cur_row, cur_col)

            if(temp_cost < dist[v]):
                dist[v] = temp_cost
                predecessor[v] = cur_node
                hq.heappush(pqueue, (dist[v], v))
    
    # total_cost = sum(dist.values())
    total_cost = 0
    for v in dist:
        total_cost = total_cost + dist[v]
    # print(total_cost)

    return total_cost


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # from collections import deque
    
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
       
    return []

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    start_pos = maze.getStart()

    visited = []

    frontier = []
    hq.heapify(frontier)
    hq.heappush(frontier, (0,0,start_pos))

    
    predecessor = dict()
    predecessor[start_pos] = None

    cost = dict()
    cost[start_pos] = 0
    
    goals = maze.getObjectives()
    goal_pos = goals[0]
    goal_row, goal_col = goal_pos

    while frontier:
        cur_f, cur_q, cur_pos = hq.heappop(frontier)
        cur_row, cur_col = cur_pos

        if cur_pos in visited:
            continue
        
        if cur_pos in maze.getObjectives():
            path = []
            while cur_pos != None:
                path.append(cur_pos)
                cur_pos = predecessor[cur_pos]
            path.reverse()
            return path

        for next_pos in maze.getNeighbors(cur_row,cur_col):
            next_row, next_col = next_pos
            temp_cost = cost[cur_pos] + m_dist(cur_row, cur_col, next_row, next_col)
            if next_pos not in cost or temp_cost < cost[next_pos]:
                cost[next_pos] = temp_cost
                q = temp_cost
                f = temp_cost + m_dist(goal_row, goal_col, next_row, next_col)
                hq.heappush(frontier,(f,q, next_pos))
                predecessor[next_pos] = cur_pos
        visited.append(cur_pos)

    return []


def find_nearest_goal(cur_node, goals):
    cur_row, cur_col = cur_node
    goal_list = []
    hq.heapify(goal_list)
    for node in goals:
        node_row, node_col = node
        dist = m_dist(node_row, node_col, cur_row, cur_col)
        hq.heappush(goal_list, (dist, node))
    cur_dist, nearest_goal = hq.heappop(goal_list)
    return cur_dist, nearest_goal

class state:
    def __init__(self, position, unfound_goals):
        self.position = position
        self.unfound_goals = unfound_goals

    def __lt__(self, other):
        return self.position < other.position


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here

    start_pos = maze.getStart()
    unfound_goals = maze.getObjectives()

    path = []
    visited = []
    path.append(start_pos)

    cur_pos = start_pos

    while unfound_goals:
        frontier = []
        hq.heapify(frontier)
        predecessor = dict()
        temp_dist, temp_nearest_goal = find_nearest_goal(cur_pos, unfound_goals)
        unfound_goals.remove(temp_nearest_goal)
        start_node = (m_dist(cur_pos[0],cur_pos[1],temp_nearest_goal[0],temp_nearest_goal[1]),0,cur_pos)
        hq.heappush(frontier,start_node)
        predecessor[cur_pos] = cur_pos

        while frontier:
            min_node = hq.heappop(frontier)
            temp, cost, cur_pos = min_node
            if cur_pos == temp_nearest_goal:
                temp_pos = cur_pos
                while predecessor[temp_pos] != temp_pos:
                    path.append(temp_pos)
                    temp_pos = predecessor[temp_pos]
                path.append(temp_pos)
                path.reverse()
                break
            
            for next_pos in maze.getNeighbors(cur_pos[0],cur_pos[1]):
                if next_pos not in predecessor:
                    hq.heappush(frontier, (cost+m_dist(next_pos[0],next_pos[1],temp_nearest_goal[0],temp_nearest_goal[1]), cost+1,next_pos))
                    predecessor[next_pos] = cur_pos
    
    return path



    # start_pos = maze.getStart()
    # unfound_goals = maze.getObjectives()

    # start_state = state(start_pos, unfound_goals)

    # visited = []

    # frontier = []
    # hq.heapify(frontier)
    # hq.heappush(frontier, (0,0,start_state))

    
    # predecessor = dict()
    # predecessor[start_state] = None

    # cost = dict()
    # cost[start_state] = 0
    # outer_counter = 0
    # while frontier: #and outer_counter < 1000:
    #     outer_counter += 1
    #     cur_f, cur_q, cur_state = hq.heappop(frontier)
    #     cur_pos = cur_state.position
    #     cur_row, cur_col = cur_pos

    #     if cur_state in visited:
    #         continue
        
    #     if cur_pos in cur_state.unfound_goals:
    #         cur_state.unfound_goals.remove(cur_pos)
    #         if len(cur_state.unfound_goals) == 0:
    #             path = []
    #             while cur_state != None:
    #                 path.append(cur_state.position)
    #                 cur_state = predecessor[cur_state]
    #             path.reverse()
    #             print(path)
    #             return path
            
    #     counter = 0
    #     for next_pos in maze.getNeighbors(cur_row,cur_col):
    #         counter += 1
    #         # if counter > 1000:
    #         #     break
    #         next_state = state(next_pos, cur_state.unfound_goals)
    #         next_row, next_col = next_pos
    #         temp_cost = cost[cur_state] + m_dist(cur_row, cur_col, next_row, next_col)
    #         if next_state not in cost or temp_cost < cost[next_state]:
    #             cost[next_state] = temp_cost
    #             q = temp_cost
    #             temp_dist, temp_nearest_goal = find_nearest_goal(next_pos, cur_state.unfound_goals)
    #             f = temp_cost + temp_dist + mst_cost(temp_nearest_goal, cur_state.unfound_goals) #temp_dist: distance from next_pos to the nearest goal
    #             hq.heappush(frontier,(f,q, next_state))
    #             predecessor[next_state] = cur_state
    #         print((next_state.position,next_state.unfound_goals,predecessor[next_state], next_state))
    #     visited.append(cur_state)
    # return []

    # start_pos = maze.getStart()
    # unfound_goals = maze.getObjectives()

    # start_state = state(start_pos, unfound_goals, [start_pos])
    

    # visited = []

    # cost = dict()
    # cost[start_state] = 0

    # predecessor = dict()
    # predecessor[start_state] = None

    # frontier = []
    # hq.heapify(frontier)
    # hq.heappush(frontier, (0, 0, len(start_state.path), start_state))

    # # print((start_state.position, start_state.unfound_goals,predecessor[start_state], start_state))

    # while frontier:
    #     cur_f, cur_g, cur_path_len, cur_state = hq.heappop(frontier)

    #     if(cur_state in visited):
    #         continue
        

    #     cur_pos = cur_state.position
    #     cur_row, cur_col = cur_pos
    #     cur_unfound_goals = cur_state.unfound_goals
    #     cur_path = cur_state.path
        
    #     if cur_pos in cur_unfound_goals:

    #         cur_unfound_goals.remove(cur_pos)
    #         # print((cur_state.position, cur_state.unfound_goals))
    #         # print(cur_unfound_goals)
    #         if(len(cur_unfound_goals) == 0):
    #             # path = []
    #             print('path')
    #             # while cur_state != None:
    #             #     print((cur_state.position, cur_state.unfound_goals, predecessor[cur_state], cur_state))
    #             #     path.append(cur_state.position)
    #             #     cur_state = predecessor[cur_state]
    #             # path.reverse()
    #             print(cur_path)
    #             return cur_path
        
    #     for next_pos in maze.getNeighbors(cur_row,cur_col):
    #         next_state = state(next_pos, cur_unfound_goals, cur_path+[next_pos])
    #         print((next_state.position,next_state.unfound_goals))
    #         print(next_state.path)
    #         next_row, next_col = next_pos
    #         temp_cost = cost[cur_state] + m_dist(cur_row, cur_col, next_row, next_col)
    #         if next_state not in cost or temp_cost < cost[next_state]:
    #             cost[next_state] = temp_cost
    #             q = temp_cost
    #             temp_dist, temp_nearest_goal = find_nearest_goal(next_pos, cur_unfound_goals)
    #             f = temp_cost + temp_dist #+ mst_cost(temp_nearest_goal, cur_unfound_goals)
    #             hq.heappush(frontier,(f,q,len(next_state.path), next_state)) 
    #             #predecessor[next_state] = cur_state
    #         # print((next_state.position,next_state.unfound_goals,predecessor[next_state], next_state))
        
    #     visited.append(cur_state)


            
        
    # start_pos = maze.getStart()
    # goals = maze.getObjectives()
    # unfound_goals = goals

    # prev_goal = None

    # frontier = []
    # hq.heapify(frontier)
    # hq.heappush(frontier, (0,0,[start_pos],unfound_goals))

    
    # predecessor = dict() # key: (pos,prev_goal) / (prev_pos)
    # predecessor[start_pos, prev_goal] = None

    # cost = dict()
    # cost[(start_pos)] = 0

    # states = dict() # key: prev_goal / visited
    # cur_goal = find_nearest_goal(start_pos, unfound_goals)
    # states[prev_goal] = [start_pos]
    # new_prev_goal = None

    # goal_found = []


    # while frontier:
    #     # print(frontier)
    #     # print(unfound_goals)
    #     cur_f, cur_q, cur_pos, prev_goal = hq.heappop(frontier)
    #     print(cur_pos)
    #     cur_row, cur_col = cur_pos
    #     if cur_pos in unfound_goals:
            
    #         unfound_goals.remove(cur_pos)
    #         if len(unfound_goals) > 0:
    #             new_prev_goal = cur_pos
    #             states[new_prev_goal] = []
    #             goal_found.append(cur_pos)
    #             cost[(cur_pos,new_prev_goal)] = 0
    #         else:
    #             path = []
    #             # print(predecessor)
    #             while prev_goal != None:
    #                 while cur_pos != None:
    #                     path.append(cur_pos)
    #                     cur_pos = predecessor[cur_pos, prev_goal]
    #                 cur_pos = prev_goal
    #                 temp, prev_goal = predecessor[-1]
    #             while cur_pos != None:
    #                 path.append(cur_pos)
    #                 cur_pos = predecessor[cur_pos, prev_goal]
    #             path.reverse()
    #             # print(path)
    #             return path

    #     for next_pos in maze.getNeighbors(cur_row,cur_col):         
    #         if next_pos in states[prev_goal]:
    #             continue

    #         next_row, next_col = next_pos
    #         # print(prev_goal)
    #         print(cur_pos)
    #         print(prev_goal)
    #         temp_cost = cost[(cur_pos,prev_goal)] + m_dist(cur_row, cur_col, next_row, next_col)
    #         if (next_pos,prev_goal) not in cost or temp_cost < cost[next_pos, prev_goal]:
    #             cost[next_pos,prev_goal] = temp_cost
    #             q = temp_cost
    #             temp_dist, temp_nearest_goal = find_nearest_goal(next_pos, unfound_goals)
    #             f = temp_cost + temp_dist + mst_cost(temp_nearest_goal, unfound_goals)
    #             # print(cur_pos)
    #             # print(goal_found)
    #             hq.heqppush(frontier,(f,q,next_pos, unfound_goals))
    #             if cur_pos in goal_found:
    #                 # print(cur_pos)
    #                 hq.heappush(frontier,(f,q, next_pos, new_prev_goal))
    #                 predecessor[next_pos, new_prev_goal] = cur_pos
    #             else:
    #                 hq.heappush(frontier,(f,q, next_pos, prev_goal))
    #                 predecessor[next_pos, prev_goal] = cur_pos
    #     if cur_pos in goal_found:
            
    #         states[new_prev_goal].append(cur_pos)

    #     else:
    #         states[prev_goal].append(cur_pos)

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
