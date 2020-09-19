
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
    angle_limits = arm.getArmLimit()

    if len(angle_limits) == 3:
        lim_alpha, lim_beta, lim_gamma = angle_limits
        offset = (lim_alpha[0],lim_beta[0], lim_gamma[0])
    elif len(angle_limits) == 2:
        lim_alpha, lim_beta= angle_limits
        lim_gamma = None
        offset = (lim_alpha[0],lim_beta[0], 0)
    elif len(angle_limits) == 1:
        lim_alpha = angle_limits[0]
        lim_beta = None
        lim_gamma = None
        offset = (lim_alpha[0],0, 0)

    num_r, num_c, num_h = 1, 1, 1
    if lim_alpha is not None:
        num_r = int((lim_alpha[1]-lim_alpha[0])/granularity) + 1

    if lim_beta is not None:
        num_c = int((lim_beta[1]-lim_beta[0])/granularity) + 1

    if lim_gamma is not None:
        num_h = int((lim_gamma[1]-lim_gamma[0])/granularity) + 1
        

    arm_angles = arm.getArmAngle()
    while len(arm_angles) < 3:
        arm_angles.append(0)
    
    start_pos = angleToIdx(arm_angles, offset, granularity)

    maze = [[[SPACE_CHAR for h in range(num_h)]for c in range(num_c)]for r in range(num_r)]

    
    for row in range(num_r):
        for col in range(num_c):
            for hei in range(num_h):
                arm_angles = idxToAngle((row,col,hei), offset, granularity)
                arm.setArmAngle(arm_angles)

                arm_tip = arm.getEnd()
                armPos = arm.getArmPos()
                armPosDist = arm.getArmPosDist()

                if doesArmTouchObjects(armPosDist, obstacles, isGoal=False) or doesArmTouchObjects(armPosDist, goals, isGoal=True) or (not isArmWithinWindow(armPos, window)):
                    if doesArmTipTouchGoals(arm_tip, goals) and (not doesArmTouchObjects(armPosDist, obstacles, isGoal=False)):
                        maze[row][col][hei] = OBJECTIVE_CHAR
                    else:
                        maze[row][col][hei] = WALL_CHAR

                prev_angles = arm_angles
    maze[start_pos[0]][start_pos[1]][start_pos[2]] = START_CHAR

    return Maze(maze, offset, granularity)