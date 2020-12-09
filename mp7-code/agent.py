import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        curr_s = self.space_mapping(state)
        if self._train is True:

            if points - self.points == 1:
                reward = 1
            elif dead:
                reward = -1
            else:
                reward = -0.1

            if self.s is not None and self.a is not None:
                self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a] += 1
                alpha = self.C / (self.C + self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a])

                a_list  = []
                for i in range(3,-1,-1):
                    a_list.append(self.Q[curr_s[0], curr_s[1], curr_s[2], curr_s[3], curr_s[4], curr_s[5], curr_s[6], curr_s[7], i])
                a_list = np.array(a_list)
                curr_a = 3 - np.argmax(a_list)
                max_Q = np.max(a_list)

                self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a] += alpha * (reward + (self.gamma * max_Q) - self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a])

            if dead:
                self.reset()
            else:
                self.s = curr_s
                self.points = points
                a_list = []
                for i in range(3,-1,-1):
                    if self.N[curr_s[0], curr_s[1], curr_s[2], curr_s[3], curr_s[4], curr_s[5], curr_s[6], curr_s[7], i] < self.Ne:
                        a_list.append(1)
                    else:
                        a_list.append(self.Q[curr_s[0], curr_s[1], curr_s[2], curr_s[3], curr_s[4], curr_s[5], curr_s[6], curr_s[7], i])
                a_list = np.array(a_list)
                self.a = 3 - np.argmax(a_list)

        else:
            self.s = curr_s
            a_list  = []
            for i in range(3,-1,-1):
                a_list.append(self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], i])
            a_list = np.array(a_list)
            # print(a_list)
            self.a = 3 - np.argmax(a_list)

        return self.a


    def space_mapping(self, state):
        # :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]

        if snake_head_x == 40:
            adjoining_wall_x = 1
        elif snake_head_x == 480:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        if snake_head_y == 40:
            adjoining_wall_y = 1
        elif snake_head_y == 480:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        if snake_head_x > food_x:
            food_dir_x = 1
        elif snake_head_x < food_x:
            food_dir_x = 2
        else:
            food_dir_x = 0

        if snake_head_y > food_y:
            food_dir_y = 1
        elif snake_head_y < food_y:
            food_dir_y = 2
        else:
            food_dir_y = 0

        if (snake_head_x, (snake_head_y - 40)) in snake_body:
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0

        if (snake_head_x, (snake_head_y + 40)) in snake_body:
            adjoining_body_bottom = 1
        else:
            adjoining_body_bottom = 0
        
        if ((snake_head_x - 40), snake_head_y) in snake_body:
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0
        
        if ((snake_head_x + 40), snake_head_y) in snake_body:
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0
        
        return [adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]