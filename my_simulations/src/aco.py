#! /usr/bin/env python

import rospy
import tf
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from math import atan2, fmod, pi
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
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
    CLASS FOR HANDLING THE MAP AND CREATING THE EDGES FOR EVERY NODE
'''
class Map:

    #class for representing the nodes
    class Nodes:        
        def __init__(self, row, col, in_map,spec):
            self.node_pos= (row, col)
            self.edges = self.compute_edges(in_map)
            self.spec = spec

        #function for creating and connecting the edges to nodes
        def compute_edges(self,map_arr):        
            imax = map_arr.shape[0]
            jmax = map_arr.shape[1]
            edges = []
            if map_arr[self.node_pos[0]][self.node_pos[1]] == 1:
                for dj in [-1,0,1]:
                    for di in [-1,0,1]:
                        newi = self.node_pos[0]+ di
                        newj = self.node_pos[1]+ dj
                        if ( dj == 0 and di == 0 or dj == 1 and di == 1 or dj == -1 and di == -1 or dj == 1 and di == -1 or dj == -1 and di == 1):
                            continue
                        if (newj>=0 and newj<jmax and newi >=0 and newi<imax):
                            if map_arr[newi][newj] == 1:
                                edges.append({'FinalNode':(newi,newj),
                                              'Pheromone': 1.0, 'Probability':
                                             0.0})
            return edges

    #initialize map nodes
    def __init__(self, maze, start, end):
        self.maze = maze
        self.initial_node = start
        self.final_node= end
        self.nodes_array = self._create_nodes()

    #create nodes out of the initial map
    def _create_nodes(self):
        return [[self.Nodes(i,j,self.maze,self.maze[i][j]) for j in
                 range(self.maze.shape[0])] for i in range(self.maze.shape[0])]
###############################################################################

'''
    CLASS FOR MAIN ACTIVITIES TO HANDLE THE ACO ALGORITHM
    HANDLING THE BEHAVIOUR OF THE WHOLE COLONY
'''
class AntColony:
    '''
        CLASS FOR HANDLING INDIVIDUAL ANT'S BEHAVIOUR
    '''
    class Ant:
        #initialize ant class and values
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node= start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []
            self.final_node_reached = False
            self.remember_visited_node(start_node_pos)

        #moving ant to selected node
        def move_ant(self, node_to_visit):          
            self.actual_node = node_to_visit
            self.remember_visited_node(node_to_visit)   

        #appends visited nodes to the list
        def remember_visited_node(self, node_pos):  
            self.visited_nodes.append(node_pos)

        #returns the list of visited nodes
        def get_visited_nodes(self):                
            return self.visited_nodes

        #checks if the ant is in final node
        def is_final_node_reached(self):            
            if self.actual_node == self.final_node :
                self.final_node_reached = True

        #uncheck the state of ant being in the final node to being able to start over
        def enable_start_new_path(self):            
            self.final_node_reached = False         

        #clears the list of visited nodes and set the first as initial
        def setup_ant(self):                        
            self.visited_nodes[1:] =[]
            self.actual_node= self.start_pos      

    #initialize antcolony class and values from input args
    def __init__(self, in_map, no_ants, iterations, evaporation_factor, pheromone_adding_constant):
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.paths = []
        self.ants = self.create_ants()
        self.best_result = []

    #creating the ants 
    def create_ants(self):                          
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.map.initial_node, self.map.final_node))
        return ants

    #randomly selecting the next node with respect to edges probability
    def select_next_node(self, actual_node):        

        #computes the total sum of the pheromone of each edge
        total_sum = 0.0
        for edge in actual_node.edges:
            total_sum += edge['Pheromone']

        #calculate probability of each edge
        prob = 0
        edges_list = []
        p = []
        for edge in actual_node.edges:
            prob = edge['Pheromone']/total_sum
            edge['Probability'] = prob
            edges_list.append(edge)
            p.append(prob)

        #clear probability values
        for edge in actual_node.edges:
            edge['Probability'] = 0.0

        #return the node based on the probability of the solutions as final node
        return np.random.choice(edges_list,1, p)[0]['FinalNode']

    #updating the pheromone levels and sorting paths by leng
    def pheromone_update(self):         

        #sorting paths based on the length
        self.sort_paths()
        for i, path in enumerate(self.paths):
            for j, element in enumerate(path):
                for edge in self.map.nodes_array[element[0]][element[1]].edges:

                    if (j+1) < len(path):
                        #if the the next node is the path's next node
                        if edge['FinalNode'] == path[j+1]:
                            edge['Pheromone'] = (1.0 -
                                                 self.evaporation_factor) * \
                            edge['Pheromone'] + \
                            self.pheromone_adding_constant/float(len(path))
                        #if the next node is not in the path's nodes
                        else:
                            edge['Pheromone'] = (1.0 -
                                                 self.evaporation_factor) * edge['Pheromone']

    #empty the list of paths
    def empty_paths(self):      
        self.paths[:]

    #sorting the paths with respect to len
    def sort_paths(self):       
        self.paths.sort(key=len)

    #appends path to the list
    def add_to_path_results(self, in_path):         
        self.paths.append(in_path)

    #coincidence indices of elements in path
    def get_coincidence_indices(self,lst, element): 
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset+1)
            except ValueError:
                return result
            result.append(offset)

    #if loops in the path -> delete them
    def delete_loops(self, in_path):    
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)

            #reverse the list to delete elements from back to front of the list
            coincidences.reverse()
            for i,coincidence in enumerate(coincidences):
                if not i == len(coincidences)-1:
                    res_path[coincidences[i+1]:coincidence] = []

        return res_path

    #handling the process of finding the best path
    def calculate_path(self):       
        for i in range(self.iterations):
            for ant in self.ants:
                ant.setup_ant()
                while not ant.final_node_reached:
                    #randomly selection of the node to visit
                    node_to_visit = self.select_next_node(self.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])])
                    #move ant to the next node randomly selected
                    ant.move_ant(node_to_visit)
                    #check if solution has been reached
                    ant.is_final_node_reached()
                #add the resulting path to the path list
                self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
                #enable the ant for a new search
                ant.enable_start_new_path()
            #update the global pheromone level
            self.pheromone_update()
            self.best_result = self.paths[0]
            #empty the list of paths
            self.empty_paths()

            #print the info about current iteration calculating
            info = "Iteration: "
            info +=  '%2s ' % str(i)
            print(info)

        #return the best path found
        return self.best_result
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
      print("Time for executing the algorithm (algorithm runtime)                     : "+str(round(t1, 4)))
      print("Time for robot to get to the final node                                  : "+str(round(t2, 4)))
      print("Overall time for robot to find the optimal path and get to the final node: "+str(round(t_all, 4)))
      print("Optimal path length                                                      : "+str(len(path) - 1))
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
    global path_done, path_index, goal, path, stop_stamp, t1, t2

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
    ap.add_argument("-p", "--p_value", required=False, type=float)
    ap.add_argument("-q", "--q_value", required=False, type=float)
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

    ants = args.ants
    iterations = args.iterations
    p = args.p_value
    q = args.q_value
    ###########################################################
    ###########################################################

    #initializing the values
    path_index = 1
    path_done = 0
    stop_stamp = 0
    goal = Point()

    #initialize node named aco for launch file
    rospy.init_node('aco')
    rospy.sleep(2)

    maze = maze[::-1]   
    maze_for_plot = np.copy(maze)
    
    maze = np.array(maze)
    maze[maze == 0] = 2
    maze[maze == 1] = 0
    maze[maze == 2] = 1
    #############################################
    #start in real coordinations
    orig_start = (int(round(curr_state[0])), int(round(curr_state[1])))      
    #####################################3#######
    #reversed coordinations for the maze array
    start = tuple(reversed(orig_start))                                         
    end = tuple(reversed(orig_end))

    #check if the start and end nodes are valid
    if maze[start[0]][start[1]] == 0 or maze[end[0]][end[1]] == 0:      
        sys.exit("Wrong initial or final node entered")

    t0 = rospy.Time.now().to_sec()      #handling the time before the algorithm starts
    
    ############################
    #handling aco processes for finding optimal path
    mapa = Map(maze, start, end)
    colony = AntColony(mapa, ants, iterations, p, q)
    path = colony.calculate_path()
    ############################

    #reverse the path for represent as the plot
    temp = []
    for i in path:
        temp.append(i[::-1])
    path = temp

    t1 = rospy.Time.now().to_sec() - t0 #handling the time for executing the algorithm to find the optimal path

    '''
        GRAPH PLOTTING
    '''
    plt.figure(figsize=(13, 13), dpi=80)
    plt.imshow(maze_for_plot, cmap='binary', interpolation = 'nearest', origin='lower')
    plt.title("Turtlebot finding the optimal path by ACO algorithm in the given map \n Setup: ants = "+str(ants)+", iterations = "+str(iterations)+", p = "+str(p)+", q = "+str(q), pad = 20, fontsize='large', fontweight='bold')
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
    #######################################################################
    #######################################################################

    while not rospy.is_shutdown():
        if path_done == 0:
            go_to_goal()
    
    rospy.spin()

if __name__ == '__main__':
    main()