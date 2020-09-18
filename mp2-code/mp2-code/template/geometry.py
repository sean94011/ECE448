# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from numpy import linalg as la
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    # Convert the angle from degree to radian
    angle_radian = math.radians(angle)

    # Get the end position with trig functions
    x = start[0] + int(length*math.cos(angle_radian))
    y = start[1] - int(length*math.sin(angle_radian))

    return (x,y)

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """

    for cur_link in armPosDist:
        start_pos = np.array(cur_link[0])
        end_pos = np.array(cur_link[1])

        end_to_start_vect = start_pos - end_pos
        start_to_end_vect = end_pos - start_pos
        
        if isGoal:
            pad_dist = 0
        else:
            pad_dist = cur_link[2]

        for cur_object in objects:
            object_pos = np.array([cur_object[0],cur_object[1]])

            end_to_obj_vect = object_pos - end_pos
            start_to_obj_vect = object_pos - start_pos

            if np.inner(end_to_start_vect,end_to_obj_vect) <= 0:
                link_obj_dist = math.sqrt(pow((end_pos[0]-object_pos[0]),2)+pow((end_pos[1]-object_pos[1]),2))
            elif np.inner(start_to_end_vect,start_to_obj_vect) <= 0:
                link_obj_dist = math.sqrt(pow((start_pos[0]-object_pos[0]),2)+pow((start_pos[1]-object_pos[1]),2))
            else:
                link_obj_dist = la.norm(np.cross(end_to_start_vect,end_to_obj_vect)) / la.norm(end_to_start_vect)

            if link_obj_dist <= cur_object[2] + pad_dist:
                    return True

    return False

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tick touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tip touches any goal. False if not.
    """

    for cur_goal in goals:
        
        if pow((cur_goal[0]-armEnd[0]),2) + pow((cur_goal[1]-armEnd[1]),2) <= pow(cur_goal[2],2):
            return True

    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    for cur_link in armPos:
        start_pos = cur_link[0]
        end_pos = cur_link[1]

        if start_pos[0] < 0 or start_pos[0] > window[0]:
            return False
        elif start_pos[1] < 0 or start_pos[1] > window[1]:
            return False
        elif end_pos[0] < 0 or end_pos[0] > window[0]:
            return False
        elif end_pos[1] < 0 or end_pos[1] > window[1]:
            return False
        else:
            continue

    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
