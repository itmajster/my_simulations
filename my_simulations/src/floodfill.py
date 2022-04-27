#! /usr/bin/env python

'''
    LOAD LIBRARIES
'''
from distutils.log import error
import pwd
import rospy
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.animation as animation
from math import pi, degrees, fmod
import time
import sys
from rospy import sleep
import tf
from math import atan2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
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
    MAIN FLOODFILL ALGORITHM FUNCTION
'''
def floodfill(maze, start, end):
    
    width = len(maze)
    height = len(maze[0])
    grid = maze
    path = []
    wall = width*height     #set the number for filling instead of wall, start, end nodes
    
    for i in range(0, len(maze)):
        for j in range(len(maze[i])):
            if grid[i][j] == 1:
                grid[i][j] = wall
                
    grid[start[0]][start[1]] = wall+1   #set values in the start and end node for number not equal to 0 because we do not want to count these nodes as the path nodes
    grid[end[0]][end[1]] = wall+2
    
    # assign the first index
    neighbors = [(start[0]-1, start[1]), (start[0]+1, start[1]), (start[0], start[1]-1), (start[0], start[1]+1)]
    for n in neighbors:
        if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1 and grid[n[0]][n[1]] != wall:
            grid[n[0]][n[1]] = 1    
                 
    # recursive function for fulfill the maze with numbers
    def fill(act_index):
        tmp = 0
        for i in range(0, len(grid)):
            if 0 in grid[i]:
                tmp = 1
        if tmp == 0:    # grid is fully marked
            return grid
        
        for x in range(0, len(grid)):
            for y in range(len(grid[x])):
                if grid[x][y] == act_index:
                    neighbors = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                    for n in neighbors:
                        if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1 and grid[n[0]][n[1]] == 0:
                            grid[n[0]][n[1]] = act_index+1
        act_index += 1
        fill(act_index)
        
    def find_path(act_node, act_index):     #if the goal is very first node from the start then return 
        if grid[act_node[0]][act_node[1]] == 1:
            path.append(start)
            return
        
        neighbors = [(act_node[0]-1, act_node[1]), (act_node[0] + 1, act_node[1]), (act_node[0], act_node[1]-1), (act_node[0], act_node[1]+1)]
        for n in neighbors:
            if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1 and grid[n[0]][n[1]] != wall and grid[n[0]][n[1]] == act_index:
                path.append((n[0], n[1]))
                break

        find_path(path[-1], act_index - 1)
                                
        
    def find_last_index():
        neighbors = [(end[0]-1, end[1]), (end[0]+1, end[1]), (end[0], end[1]-1), (end[0], end[1]+1)]
        for n in neighbors:
            if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1 and grid[n[0]][n[1]] != wall:
                point_index = grid[n[0]][n[1]]
                break
                
        for n in neighbors:
            if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1 and grid[n[0]][n[1]] != wall:
                if grid[n[0]][n[1]] < point_index:
                    point_index = grid[n[0]][n[1]]
                    
        return point_index
                    
                    
    act_index = 1                   
    fill(act_index)                 #fulfill the maze with numbers
    
    path.append(end)                #start append the path but from behind with end node
    act_index = find_last_index()   #find the actual index to start making the path
    find_path(end, act_index)       #append nodes to the path array from end node according to numbers ascending
    maze[start[0]][start[1]] = 0
    maze[end[0]][end[1]] = 0
    return path[::-1]               #return path array for the robot but furst reverse the array...because we filled array the opposite way
    print(path)
################################################################################################################
################################################################################################################

'''
    CALLBACK FUNCTION FOR ACTUALIZING A POSITION AND AN ANGLE OF THE ROBOT
'''
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
################################################################################################################
################################################################################################################

    

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
    global path_done, path_index, stop_stamp, t1, t2, path
    rotate_speed = 0.5
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
    global path_done, path_index, goal, path, stop_stamp, t1, t2   #global variables

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

    rospy.init_node('floodfill')        #initialize the node named floodfill...for launch file
    rospy.sleep(2)


    tmp_maze = maze[::-1]    
    #############################################3
    #real coordinates
    orig_start = (int(round(curr_state[0])), int(round(curr_state[1])))      
    #####################################3#######
    #array coordinates as reversed real coordinates
    start = tuple(reversed(orig_start))  
    end = tuple(reversed(orig_end))

    #check if start or end node is either in the map or valid empty node
    if tmp_maze[start[0]][start[1]] == 1 or tmp_maze[end[0]][end[1]] == 1: 
        sys.exit("Wrong initial or final node entered")

    t0 = rospy.Time.now().to_sec()      #handling the time before the algorithm starts

    #############################
    #call function for resulting the final optimal path
    orig_path = floodfill(tmp_maze, start, end)
    #############################

    #reverse the path for represent as the plot
    temp = []
    for i in orig_path:
        temp.append(i[::-1])
    path = temp

    t1 = rospy.Time.now().to_sec() - t0                  #handling the time for executing the algorithm to find the optimal path

    '''
        GRAPH PLOTTING
    '''
    plt.figure(figsize=(13, 13), dpi=80)
    plt.imshow(tmp_maze, cmap='binary', interpolation = 'nearest', origin='lower')
    plt.title("Turtlebot finding the optimal path by floodfill algorithm in the given map", pad = 20, fontsize='large', fontweight='bold')
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