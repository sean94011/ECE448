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

    graph = objectives[:]

    for vertex in graph:
        dist[vertex] = float('inf')
        predecessor[vertex] = None

    dist[start_node] = 0
    
    pqueue = []
    hq.heapify(pqueue)
    hq.heappush(pqueue, (dist[start_node],start_node))

    while pqueue:
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
            temp_cost = m_dist(cur_row, cur_col, v_row, v_col)
            if(temp_cost < dist[v]):
                dist[v] = temp_cost
                predecessor[v] = cur_node
                hq.heappush(pqueue, (dist[v], v))
    
    total_cost = 0
    for v in dist:
        total_cost = total_cost + dist[v]

    return total_cost

def mst_cost_multi(edge_weight, start_node, objectives):
    dist = dict()
    predecessor = dict()
    visited = []

    graph = objectives[:]

    for vertex in graph:
        dist[vertex] = float('inf')
        predecessor[vertex] = None

    dist[start_node] = 0
    
    pqueue = []
    hq.heapify(pqueue)
    hq.heappush(pqueue, (dist[start_node],start_node))

    while pqueue:
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
            temp_cost = edge_weight[cur_node,v]
            if(temp_cost < dist[v]):
                dist[v] = temp_cost
                predecessor[v] = cur_node
                hq.heappush(pqueue, (dist[v], v))
    

    total_cost = 0
    for v in dist:
        total_cost = total_cost + dist[v]
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
                g = temp_cost
                f = temp_cost + m_dist(goal_row, goal_col, next_row, next_col)
                hq.heappush(frontier,(f,g, next_pos))
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
    def __init__(self, position, unfound_goals, path):
        self.position = position
        self.unfound_goals = unfound_goals
        self.path = path


    def __lt__(self, other):
        return len(self.unfound_goals) < len(other.unfound_goals)

class state_multi:
    def __init__(self, position, unfound_goals):
        self.position = position
        self.unfound_goals = unfound_goals


    def __lt__(self, other):
        return len(self.unfound_goals) < len(other.unfound_goals)



def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here

    start_pos = maze.getStart()
    unfound_goals = maze.getObjectives()

    start_state = state(start_pos, unfound_goals, [start_pos])

    visited = []

    frontier = []
    hq.heapify(frontier)
    hq.heappush(frontier, (0,0,start_state))

    closed = []

    cost = dict()
    cost[(start_state.position, tuple(start_state.unfound_goals))] = 0

    while frontier:
        cur_f, cur_q, cur_state = hq.heappop(frontier)
        cur_pos = cur_state.position
        cur_row, cur_col = cur_pos
        cur_unfound_goals = []
        for i in cur_state.unfound_goals:
            cur_unfound_goals.append(i)

        if (cur_state.position,cur_state.unfound_goals) in visited:
            continue
        visited.append((cur_state.position,cur_state.unfound_goals))
        
        if cur_pos in cur_state.unfound_goals:
            cur_unfound_goals.remove(cur_pos)
            if len(cur_unfound_goals) == 0:
                return cur_state.path
            
        for next_pos in maze.getNeighbors(cur_row,cur_col):
            next_state = state(next_pos, cur_unfound_goals, cur_state.path+[next_pos])
            next_row, next_col = next_pos
            temp_cost = cost[(cur_state.position,tuple(cur_state.unfound_goals))] + m_dist(cur_row, cur_col, next_row, next_col)
            if (next_state.position,tuple(next_state.unfound_goals)) not in cost or temp_cost < cost[(next_state.position,tuple(next_state.unfound_goals))]:
                cost[(next_state.position,tuple(next_state.unfound_goals))] = temp_cost
                g = temp_cost
                temp_dist, temp_nearest_goal = find_nearest_goal(next_pos, cur_unfound_goals)
                f = temp_cost + temp_dist + mst_cost(temp_nearest_goal, cur_unfound_goals) #temp_dist: distance from next_pos to the nearest goal
                hq.heappush(frontier,(f,g, next_state))

    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    start_pos = maze.getStart()
    unfound_goals = maze.getObjectives()


    temp_goals = maze.getObjectives()

    mst = dict()
    nearest_neighbor = dict()
    
    

    start_state = state_multi(start_pos, unfound_goals)

    visited = []
    predecessor = dict()

    frontier = []
    hq.heapify(frontier)
    hq.heappush(frontier, (0,0,start_state))

    predecessor[(start_pos,tuple(unfound_goals))] = None

    edge_weight = dict()
    for i in unfound_goals:
        for j in unfound_goals:
            edge_weight[i,j] = len(astar_fast(maze,i,j))

    cost = dict()
    cost[(start_state.position, tuple(start_state.unfound_goals))] = 0
    outer_counter = 0
    while frontier:

        cur_f, cur_q, cur_state = hq.heappop(frontier)
        cur_pos = cur_state.position
        cur_row, cur_col = cur_pos
        cur_unfound_goals = cur_state.unfound_goals[:]

        if (cur_state.position,tuple(cur_unfound_goals)) in visited:
            continue
        visited.append((cur_state.position,tuple(cur_unfound_goals)))

        if cur_pos in cur_unfound_goals:
            cur_unfound_goals.remove(cur_pos)
            if len(cur_unfound_goals) == 0:
                path = []
                cur_unfound_goals.append(cur_pos)
                temp_cur_unfound_goals = tuple(cur_unfound_goals)
                temp_tuple = (cur_pos, temp_cur_unfound_goals)
                while temp_tuple != None:
                    path.append(temp_tuple[0])
                    temp_tuple = predecessor[temp_tuple]
                path.reverse()
                return path

        for next_pos in maze.getNeighbors(cur_row,cur_col):
            next_state = state_multi(next_pos, cur_unfound_goals)
            if (next_state.position, tuple(next_state.unfound_goals)) in visited:
                continue
            next_row, next_col = next_pos

            temp_cost = cost[(cur_state.position,tuple(cur_state.unfound_goals))] + m_dist(cur_row, cur_col, next_row, next_col)
            temp_next_unfound_goals = next_state.unfound_goals
            if (next_state.position,tuple(temp_next_unfound_goals)) not in cost or temp_cost < cost[(next_state.position,tuple(temp_next_unfound_goals))]:
                cost[(next_state.position,tuple(temp_next_unfound_goals))] = temp_cost
                g = temp_cost
                temp_dist, temp_nearest_goal = find_nearest_goal(next_pos, cur_unfound_goals)
                if (tuple(cur_unfound_goals)) not in mst:
                    mst[tuple(cur_unfound_goals)] = mst_cost_multi(edge_weight, cur_unfound_goals[0], cur_unfound_goals)
                cur_mst_cost = mst[tuple(cur_unfound_goals)]
                f = temp_cost + temp_dist + cur_mst_cost
                hq.heappush(frontier,(f,g, next_state))
                predecessor[(next_state.position, tuple(next_state.unfound_goals))] = (cur_state.position ,tuple(cur_state.unfound_goals))
    return []

def build_mst(maze, start_node, objectives):
    dist = dict()
    predecessor = dict()
    visited = []

    graph = objectives + [start_node]

    for vertex in graph:
        dist[vertex] = float("inf")
        predecessor[vertex] = None

    dist[start_node] = 0
    
    pqueue = []
    hq.heapify(pqueue)
    hq.heappush(pqueue, (dist[start_node],start_node))

    while pqueue:
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
            temp_cost = len(astar_fast(maze, v, cur_node))

            if(temp_cost < dist[v]):
                dist[v] = temp_cost
                predecessor[v] = cur_node
                hq.heappush(pqueue, (dist[v], v))

    mst = dict()
    for vertex in graph:
        mst[vertex] = []
    for vertex in graph:
        if predecessor[vertex] not in mst:
            mst[predecessor[vertex]] = []
        mst[predecessor[vertex]].append(vertex)

    return mst


def astar_fast(maze, start, goal):

    start_pos = start

    visited = []

    frontier = []
    hq.heapify(frontier)
    hq.heappush(frontier, (0,0,start_pos))

    
    predecessor = dict()
    predecessor[start_pos] = None

    cost = dict()
    cost[start_pos] = 0
    
    goal_pos = goal
    goal_row, goal_col = goal_pos

    while frontier:
        cur_f, cur_q, cur_pos = hq.heappop(frontier)
        cur_row, cur_col = cur_pos

        if cur_pos in visited:
            continue
        
        if cur_pos == goal_pos:
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
                g = temp_cost
                f = temp_cost + m_dist(goal_row, goal_col, next_row, next_col)
                hq.heappush(frontier,(f,g, next_pos))
                predecessor[next_pos] = cur_pos
        visited.append(cur_pos)

    return []

def traverse(mst, path, cur_pos):
    path.append(cur_pos)
    if len(mst[cur_pos]) == 0:
        return
    for child in mst[cur_pos]:
        traverse(mst,path,child)
        path.append(cur_pos)


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start_pos = maze.getStart()
    objectives = maze.getObjectives()

    cur_mst = build_mst(maze,start_pos,objectives)

    graph = [start_pos] + objectives
    path_between_dots = dict()
    for i in graph:
        for j in graph:
            if i == j:
                continue
            temp_path = astar_fast(maze, i, j)
            temp_path.remove(temp_path[len(temp_path)-1])
            path_between_dots[(i,j)] = temp_path
    temp_graph_path = []
    traverse(cur_mst, temp_graph_path,start_pos)

    graph_path = []
    for node in temp_graph_path:
        graph_path.append(node)
        if node in graph:
            graph.remove(node)
        if len(graph) == 0:
            break

    fast_path = []
    for i in range(len(graph_path)-1):
        fast_path.extend(path_between_dots[(graph_path[i],graph_path[i+1])])
        
    fast_path.append(graph_path[-1])
    return fast_path
