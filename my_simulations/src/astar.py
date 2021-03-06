#! /usr/bin/env python

import rospy
import tf
import numpy as np
from math import atan2, fmod, pi
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import heapq
import argparse
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
    A node class for A* Pathfinding
'''
class Node():

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

'''
    Main function for handling the path finding algorithm and returns a list of tuples as a path from the given start to the given end in the given maze
'''

#heuristic function to calculate distance between nodes a and b
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(map, start, goal):
    #neighbours for every node (4 way movement --- up, down, right, left)
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

    #close list and g(), f() for cost calculations
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            data = []

            while current in came_from:
                data.append(current)
                current = came_from[current]

            return data
        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < map.shape[0]:
                if 0 <= neighbor[1] < map.shape[1]:                
                    if map[neighbor[0]][neighbor[1]] == 1:
                        continue

                else:
                    #if array bounds walls in y axis
                    continue

            else:
                #if array bounds walls in x axis
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return False


################################################################
################################################################
################################################################
'''
def astar(maze, start, end):

    #create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    #initialize open and closed list
    open_list = []
    closed_list = []

    #add the start node
    open_list.append(start_node)

    #loop until the end is found
    while len(open_list) > 0:

        #get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        #pop current node off the open list and add node to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        #found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        #children array as the all of neighbouring nodes
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            #check if node is empty
            if maze[node_position[0]][node_position[1]] != 0:
                continue
            # check if valid (not neccessary bcs of the side walls)
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        #looping children nodes
        for child in children:
            #check if the node is on closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue
            #computing of the cost funkction f() = g() + h()
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # if childis not in closed open list to the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue
            open_list.append(child)
'''
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

def stop_robot():
    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    vel_pub.publish(msg) 

def go_to_goal():
    global path_done, path_index, stop_stamp, t1
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
      plt.title("Turtlebot finding the optimal path by A* algorithm in the given map", pad = 20, fontsize='large', fontweight='bold')
      plt.show()
      path_done = 1
      pass

################################################################################################################
################################################################################################################

'''
	SUBSCRIBER FOR HANDLING ODOMETRY TOPIC TO COMPUTE CURRENT ANGLE AND POSITION OF THE ROBOT
	PUBLISHER FOR PUBLISHING VELOCITY OF THE ROBOT
'''
rospy.Subscriber('/odom', Odometry, callback)               #subscriber for the odometry
vel_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1) #publisher for the velocity

def main():
    global path_done, path_index, goal, move, path, stop_stamp, t1

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
    ###########################################################
    ###########################################################

    path_index = 1
    path_done = 0
    stop_stamp = 0
    goal = Point()

    move = Twist()
    move.linear.x = 0.0
    move.linear.y = 0.0
    move.linear.z = 0.0
    move.angular.x = 0.0
    move.angular.y = 0.0
    move.angular.z = 0.0

    #initialize node named astar for launch file
    rospy.init_node('astar')
    rospy.sleep(2)

    tmp_maze = maze[::-1]    
    #############################################3
    #start in real coordinations
    orig_start = (int(round(curr_state[0])), int(round(curr_state[1])))      
    #####################################3#######
    #reversed coordinations for the maze array
    start = tuple(reversed(orig_start))  
    end = tuple(reversed(orig_end))

    #check if start or end node is either in the map or valid empty node
    if tmp_maze[start[0]][start[1]] == 1 or tmp_maze[end[0]][end[1]] == 1:
        sys.exit("Wrong initial or final node entered")

    t0 = rospy.Time.now().to_sec()      #handling the time before the algorithm starts

    ################################
    #astar function calling to return optimal path
    #path = astar(tmp_maze, start, end)
    path = astar(tmp_maze, start, end)
    path.append(start)
    path = path[::-1]
    ################################

    #reverse the path for represent as the plot
    temp = []
    for i in path:
        temp.append(i[::-1])
    path = temp

    t1 = rospy.Time.now().to_sec() - t0                  #handling the time for executing the algorithm to find the optimal path

    '''
        GRAPH PLOTTING
    '''
    plt.figure(figsize=(13, 13), dpi=80)
    plt.imshow(tmp_maze, cmap='binary', interpolation = 'nearest', origin='lower')
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