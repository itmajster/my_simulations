#! /usr/bin/env python

import rospy
import tf
import random
import math
import sys
import os
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from math import atan2, fmod, pi
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
################################################################################################################
################################################################################################################

'''
    LOAD IMAGE AND TEXT FILE AS NUMPY ARRAY
'''
################################################################################################################
################################################################################################################

'''
    LOAD IMAGE AND TEXT FILE AS NUMPY ARRAY
'''
#load image file to numpy array
def image_load(x_size, y_size, file_name):  
	username = os.getlogin()
	img_path = '/home/'+username+'/catkin_ws/src/my_simulations/imgs/'+file_name
	image = Image.open(img_path)
	image = image.convert("1")
	size = image.size
	pixels = image.load()
	maze_image = []

	# maze size. uploaded as input arg by user
	os_x = x_size 
	os_y = y_size

	#compute size in pixels of one section (node)
	grids_x = size[0]/os_x  
	grids_y = size[1]/os_y

	#array for positions of pixels (nodes)
	x_array = []    
	y_array = []

	#first position
	x_array.append(grids_x / 2)	
	y_array.append(grids_y / 2)

	#load indexes of other positions
	for i in range (1, os_x):
		x_array.append(x_array[i-1] + grids_x)
		
	for i in range (1, os_y):
		y_array.append(y_array[i-1] + grids_x)

	#Obstacle (1) and empty node(0) according to brughtness of the pixel
	for y in range(os_y):		
		tmp = []
		for x in range(os_x):
			if pixels[x_array[x], y_array[y]] == 0: tmp.append(1)
			else:   tmp.append(0)
		maze_image.append(tmp)

	return np.array(maze_image)

#load text file to numpy array
def txt_load(file_name):                
	username = os.getlogin()
	txt_path = '/home/'+username+'/catkin_ws/src/my_simulations/txts/'+file_name
	print(txt_path)
	raw_map = np.loadtxt(txt_path, dtype = 'int')
	return np.array(raw_map)
################################################################################################################
################################################################################################################

'''
    POPULATION INITIALIZING
'''
class Population:
    def __init__(self):
        self.path = []
        self.fitness = 0
        self.percent = 0

'''
    GENETIC ALGORITHM CLASS
'''
class Genetic:
	#pop_size: population size
	#leng_ind: length of individual
	#mut_prob: probability of mutation
	#move: directions 0 to 3: left, up, right, down
    
    def __init__(self, pop_size, mut_prob, maze):
        self.pop_percentages = []
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.maze = maze
        self.maze_size = maze.shape[0]
        self.population = []
        self.move = [ [0, -1], [-1, 0], [0, 1], [1, 0]]
        self.leng_ind = int(self.maze_size*2)
        print("Leng_ind: %d" % self.leng_ind)
        
    def create_population(self):
        self.population = []
        for i in range(self.pop_size):
            pop_tmp = Population()
            for j in range(self.leng_ind):
                pop_tmp.path.append(random.randint(0, 3))
            self.population.append(pop_tmp)
            
    def genetic_var(self):
        self.find_fitnesses()
        self.assign_percents()
        times = 5000
        counter = 0

        #start the counter for 5000 generations
        while counter < times:
            new_pop = []

            #if the path diverged create new population and recompute the fitnesses and percentages
            if (self.pop_diverged()):
                self.create_population()
                self.find_fitnesses_var()
                self.assign_percents()

            #create new population by reproducing based on the selection
            for i in range(self.pop_size):
                x = self.random_selection()
                y = self.random_selection()
                child = self.reproduce_var(x, y)

                #according to thw mutation probability mutate children
                if (random.randint(1, 10) <= self.mut_prob):
                    child = self.mutate(child)

                #append the child
                new_pop.append(child)

            # calculate fitnesses and percentages of new population 
            # and find the best one
            self.population = new_pop
            self.find_fitnesses_var()
            self.assign_percents()
            pop = self.get_best_ind()

            #if the optimal solution is found return the individual (path)
            if pop.fitness == self.maze_size:
                info = "Best Fitnes: "
                info +=  '%2s ' % str(int(pop.fitness))
                info += ' Length of Ind: %2s' % str(len(pop.path))
                info += ' %s %5s %s %s' % (str("Step: "), str(counter), str("/"), str(times))
                info += " -"
                info += ">" * int(counter / 200)
                print(info)
                print("Found!!!!!")
                return pop

            #if the path is too long and not optimal, create new population
            if len(pop.path) > 5 * self.maze_size:
                self.create_population()
            info = "Best Fitnes: "
            info +=  '%2s ' % str(int(pop.fitness))
            info += ' Length of Ind: %3s' % str(len(pop.path))
            info += '  %s %5s %s %s' % (str("Step: "), str(counter), str("/"), str(times))
            info += " -"
            info += ">" * int(counter / 200)

            #if the counter is converges to the end, display the info
            if counter == times - 1 :
                print(info)
            else:
                print(info, end="\r")
            counter += 1

        #after 5000 generations or if the best solution is found 
        #return the best path as result
        if counter >= times:
            print("Individual not found")
        return self.get_best_ind()

	#find if the individual diverged
    def pop_diverged(self):
        for ind in self.population:
            if len(ind.path) >= self.maze_size - 5 :
                return False
        return True

	
	#creates an array containing number of individuals.
	#an individual takes place in the array with respect to its fitness.
	
    #calculating percentage fitness of the individual with respect to its population
    def create_pop_percentages(self):
        self.pop_percentages = []
        for individual in self.population:
            for i in range(individual.percent):
                self.pop_percentages.append(individual)
		
    #find the best individual by fitness of percentages
    def get_best_ind(self):
        best_ind = self.population[0]
        for i in range(1, self.pop_size):
            if self.population[i].fitness > best_ind.fitness:
                best_ind = self.population[i]
        return best_ind

    #random selection with respect to its fitnesses
    def random_selection(self):
        if len(self.pop_percentages) == 0:
            return self.population[random.randint(0, self.pop_size-1)]
        return self.pop_percentages[random.randint(0, len(self.pop_percentages)-1)]

    #reproducing the new individual based on the parents genes
    def reproduce_var(self, ind1, ind2):
        new_child = Population()
        if (len(ind1.path) <= 1):
            new_child.path = deepcopy(ind2.path)
            return new_child

        loc = random.randint(0, len(ind1.path)-1)

        for i in range(0, loc):
            new_child.path.append(ind1.path[i])

        for i in range(loc, len(ind2.path)):
            new_child.path.append(ind2.path[i])

        for i in range(loc, len(ind1.path)):
            new_child.path.append(ind1.path[i])

        return new_child

    #mutate individual by randomly replacing gene by random gene
    def mutate(self, ind):
        if (len(ind.path) <=  1):
            return ind
        loc = random.randint(0, len(ind.path)-1)
        ind.path[loc] = random.randint(0, 3)
        return ind

    #find fitness (pytagorian distance to end node)
    #fitness = maze_size - individual's distance from target to its last point in the path
    def find_fitnesses(self):
        blocked = False
        for individual in self.population:
            blocked = False
            current_point =  start		
            for direct in individual.path:
                tmp = [current_point[0] + self.move[direct][0], current_point[1] + self.move[direct][1]]
                if self.is_blocked(tmp):
                    individual.fitness = 0
                    blocked = True
                    break
                current_point = tmp
            if not blocked:
                individual.fitness = self.maze_size - self.get_dist(current_point)
	
	#individual fitness = maze_size - individual's distance from target to its closest point to target
    def find_fitnesses_var(self):
        global start

        for individual in self.population:
            blocked = False
            current_point =  start	
            min_dist = self.get_dist(current_point)
            index = 0
            new_path = []

            #find if the node is blocked or not
            for i, direct in enumerate(individual.path):
                tmp = [current_point[0] + self.move[direct][0], current_point[1] + self.move[direct][1]]
                if self.is_blocked(tmp):
                    individual.fitness = 0
                    blocked = True
                    break

                current_point = tmp
                tmp_dist = self.get_dist(current_point)
                if tmp_dist < min_dist:
                    min_dist = tmp_dist
                    index = i
            for j in range(index+1):
                new_path.append(individual.path[j])
            individual.path = new_path

            #if node is not blocked -> calculate distance fitness
            if not blocked:
                individual.fitness = self.maze_size - self.get_dist(current_point)

    #calculate pytagorian fitness between desired node and end node (c**2 = a**2 + b**2)
    def get_dist(self, point):
        global end
        return math.sqrt((end[0] - point[0])**2 + (end[1] - point[1])**2)

    #find if the desired location/node is either empty or blocked by obstacle
    def is_blocked(self, location):
        if (location[0] < 0 or location[1] < 0 or location[0] >= self.maze.shape[0] or location[1] >= self.maze.shape[1]):
            return True
        if (self.maze[location[0]][location[1]] == 1):
            return True
        return False

    #calculate percen and assign it to every individual
    def assign_percents(self):
        total = 0
        for individual in self.population:
            total += individual.fitness
        for individual in self.population:
            if individual.fitness != 0 and total != 0:
                individual.percent = int(1 + individual.fitness  * 100 / total)
            else:
                individual.percent = 0
        total = 0
        for individual in self.population:
            total += individual.percent

        self.create_pop_percentages()

################################################################################################################
################################################################################################################

def callback(msg):
    global curr_state

    quaternion = (
    msg.pose.pose.orientation.x,
    msg.pose.pose.orientation.y,
    msg.pose.pose.orientation.z,
    msg.pose.pose.orientation.w)

    euler = tf.transformations.euler_from_quaternion(quaternion)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    curr_state = [msg.pose.pose.position.x,msg.pose.pose.position.y,yaw]

def go_to_goal():
    global path_done, path_index, stop_stamp, t1, t2, path
    rotate_speed = 1.0
    forward_speed = 0.15
    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0     

    if path_index < len(path): # finding for next Point to move in
        goal.x = path[path_index][0] 
        goal.y = path[path_index][1] 
    else: # robot is in the goal
        stop_stamp = 1
    
    if(stop_stamp == 0):
        inc_x = goal.x - curr_state[0]
        inc_y = goal.y - curr_state[1]

        angle_to_goal = atan2(inc_y, inc_x)
        two_point_distance_to_goal = np.sqrt(inc_x*inc_x + inc_y*inc_y)
        theta = curr_state[2]
        anglediff = fmod((angle_to_goal - theta), 2*pi)

        if (anglediff < 0.0):
            if (abs(anglediff) > (2*pi + anglediff)):
                anglediff = 2*pi + anglediff
        else:
            if (anglediff > abs(anglediff - 2*pi)):
                anglediff = anglediff - 2*pi

        if(two_point_distance_to_goal >= 0.05):
            if abs(anglediff) > 0.1:
                msg.linear.x = 0.0
                msg.angular.z = anglediff
                vel_pub.publish(msg)
            else:
                msg.linear.x = min(forward_speed, two_point_distance_to_goal)
                msg.angular.z = 0.0
                vel_pub.publish(msg)
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            vel_pub.publish(msg)
            path_index += 1
    
    else:
      print("FINALLY HAVE ARRIVED TO MY DESTINATION!!!!HOORAAAAY")
      t2 = rospy.Time.now().to_sec() - t1            #handling the time for robot to get to tha target
      t_all = t1 + t2
      print("Time for executing the algorithm (algorithm runtime)                     : "+str(round(t1, 6)))
      print("Time for robot to get to the final node                                  : "+str(round(t2, 6)))
      print("Overall time for robot to find the optimal path and get to the final node: "+str(round(t_all, 6)))
      print("Optimal path length                                                      : "+str(len(path)-1))
      plt.show()
      path_done = 1
      pass

'''
	SUBSCRIBER FOR HANDLING ODOMETRY TOPIC TO COMPUTE CURRENT ANGLE AND POSITION OF THE ROBOT
	PUBLISHER FOR PUBLISHING VELOCITY OF THE ROBOT
'''
rospy.Subscriber('/odom', Odometry, callback)               #subscriber for the odometry
vel_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1) #publisher for the velocity

def main():
    global path_done, path_index, goal, move, path, stop_stamp, start, end, t1, t2

    '''
        HANDLING THE ARGUMENTS
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-fx", "--final_x", required=True, type=int)
    ap.add_argument("-fy", "--final_y", required=True, type=int)
    ap.add_argument("-img", "--image", required=False, type=str)
    ap.add_argument("-szx", "--sizex", required=False, type=int)
    ap.add_argument("-szy", "--sizey", required=False, type=int)
    ap.add_argument("-txt", "--text", required=False, type=str)
    ap.add_argument("-ans", "--ants", required=False, type=int)
    ap.add_argument("-its", "--iterations", required=False, type=int)
    ap.add_argument("-psz", "--pop_size", required=False, type=float)
    ap.add_argument("-prb", "--mut_prob", required=False, type=float)
    args, unknown = ap.parse_known_args()

    if args.image == "NULL" and args.text == "NULL":
        sys.exit("No map value was entered. Remember you can enter the image name or text file name for loading the map.")
    elif args.image != "NULL":
        image_name = args.image
        if args.sizex == 0 or args.sizey == 0:
            sys.exit("no x or y axis size value entered")
        else:
            axisx = args.sizex
            axisy = args.sizey
            maze = image_load(axisx, axisy, image_name)
    elif args.text != "NULL":
        text_name = args.text
        maze = txt_load(text_name)

    if len(maze[0])-1 > args.final_x > 0 and len(maze)-1 > args.final_y > 0:
        orig_end = (args.final_x, args.final_y)
    else:
        sys.exit("Wrong final node position entered")

    pop_size = int(args.pop_size)
    mut_prob = args.mut_prob   # 1-10    higher<->lower
    ###########################################################
    ###########################################################

    #initialize variables
    path_index = 1
    path_done = 0
    stop_stamp = 0
    goal = Point()

    #initialize node named ga for launch file
    rospy.init_node('ga')
    rospy.sleep(2)

    #get array of maze
    maze = np.array(maze)
    print(maze.shape[0])
    
    #return maze for operating at nodes
    maze = maze[::-1]   

    ##################
    #start in real coordinations
    orig_start = (int(round(curr_state[0])), int(round(curr_state[1]))) 
    ##################
    #reversed coordinations for the maze array
    start = tuple(reversed(orig_start))     
    end = tuple(reversed(orig_end))

    #check if the start and end nodes are valid
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:  
        sys.exit("Wrong initial or final node entered")

    t0 = rospy.Time.now().to_sec()      #handling the time before executing algorithm

    #######################################
    #compute the optimal path in the given maze
    #according to GA processes
    gen = Genetic(pop_size, mut_prob, maze)
    gen.create_population()
    gen.find_fitnesses()
    gen.assign_percents()
    result = gen.genetic_var()
    #######################################

    #show the resulted individual as the movements numbers
    print("Result: {0}".format(result.path))
    
    #create the array of nodes as the final path from resulted individual
    next_move = [[0, -1], [-1, 0], [0, 1], [1, 0]]
    path = []
    path.append(list(start))
    for i in result.path:
        act_node = path[-1]
        path.append([act_node[0] + next_move[i][0], act_node[1] + next_move[i][1]])
    
    #remove dead loops in the resulted path
    for j in range(3):
        for i in range(1, len(path)-1):
            if path[i] == path[-1]:
                break
            if path[i-1] == path[i+1]:
                del path[i]
                del path[i]

    #convert path array to desired form
    temp = []
    for i in path:
        temp.append(i[::-1])
    path = temp
    print(path)
    print(len(path))

    t1 = rospy.Time.now().to_sec() - t0 #handling the time for executing the algorithm to find the optimal path

    '''
        GRAPH PLOTTING
    '''
    plt.figure(figsize=(13, 13), dpi=80)
    plt.imshow(maze, cmap='binary', interpolation = 'nearest', origin='lower')
    plt.title("Turtlebot finding the optimal path by Genetic algorithm in the given map \n Setup: population size = "+str(pop_size)+", mutation probability = "+str((10-mut_prob)*10)+"%", pad = 20, fontsize='large', fontweight='bold')
    plt.xlabel("X axis of the map", fontsize = 'large')
    plt.ylabel("Y axis of the map", fontsize = 'large')

    listOf_Xticks = np.arange(0, len(maze), 1)
    plt.xticks(listOf_Xticks)       #setting the interval of x-axis
    listOf_Yticks = np.arange(0, len(maze[0]), 1)
    plt.yticks(listOf_Yticks)       #setting the interval of y-axis

    #plt.grid()
    plt.plot([i[0] for i in path], [i[1] for i in path], label="optimal path")
    plt.plot(orig_start[0], orig_start[1], 'ro', markersize=15, label="start node")
    plt.plot(orig_end[0], orig_end[1], 'bo', markersize=15, label="end node")
    plt.legend()
    #plt.show()
    #######################################################################
    #######################################################################

    while not rospy.is_shutdown():
        if path_done == 0:
            go_to_goal()

    rospy.spin()

if __name__ == '__main__':
    main()