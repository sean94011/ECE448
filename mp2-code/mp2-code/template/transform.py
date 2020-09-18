
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    lim_alpha, lim_beta = arm.getArmLimit()

    num_row = int((lim_alpha[1]-lim_alpha[0])/granularity) + 1
    num_col = int((lim_beta[1]-lim_beta[0])/granularity) + 1

    offset = (lim_alpha[0],lim_beta[0])

    start_pos = angleToIdx(arm.getArmAngle(), offset, granularity)

    maze = [[SPACE_CHAR for col in range(num_col)]for row in range(num_row)]


    for row in range(num_row):
        for col in range(num_col):
            arm_angles = idxToAngle((row,col), offset, granularity)
            arm.setArmAngle(arm_angles)

            arm_tip = arm.getEnd()
            armPos = arm.getArmPos()
            armPosDist = arm.getArmPosDist()

            if doesArmTouchObjects(armPosDist, obstacles, isGoal=False) or doesArmTouchObjects(armPosDist, goals, isGoal=True) or (not isArmWithinWindow(armPos, window)):
                if doesArmTipTouchGoals(arm_tip, goals):
                    maze[row][col] = OBJECTIVE_CHAR
                else:
                    maze[row][col] = WALL_CHAR
    
    maze[start_pos[0]][start_pos[1]] = START_CHAR

    return Maze(maze, offset, granularity)