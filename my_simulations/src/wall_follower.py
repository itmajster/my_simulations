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
from math import fmod, pi, degrees
import time
import sys
from rospy import sleep
import tf
from tf.transformations import euler_from_quaternion
from math import atan2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
################################################################################################################
################################################################################################################

'''
    LOAD IMAGE AND TEXT FILE AS NUMPY ARRAY
'''
def image_load(x_size, y_size, file_name):  #load image file to numpy array
	username = os.getlogin()
	img_path = '/home/'+username+'/catkin_ws/src/my_simulations/imgs/'+file_name
	image = Image.open(img_path)
	image = image.convert("1")
	size = image.size
	pixels = image.load()
	maze_image = []

	os_x = x_size # rozmer mapy, zadava uzivatel
	os_y = y_size

	start = (1, 1)
	end = (8, 8)

	start = tuple(reversed(start))
	end = tuple(reversed(end))

	grids_x = size[0]/os_x  #velkost pixelov jedneho uzla
	grids_y = size[1]/os_y

	x_array = []    #suradnice uzlov
	y_array = []

	x_array.append(grids_x / 2)
	y_array.append(grids_y / 2)

	for i in range (1, os_x):
		x_array.append(x_array[i-1] + grids_x)
		
	for i in range (1, os_y):
		y_array.append(y_array[i-1] + grids_x)

	for y in range(os_y):
		tmp = []
		for x in range(os_x):
			if pixels[x_array[x], y_array[y]] == 0: tmp.append(1)
			else:   tmp.append(0)
		maze_image.append(tmp)

	return np.array(maze_image)

def txt_load(file_name):                #load text file to numpy array
	username = os.getlogin()
	txt_path = '/home/'+username+'/catkin_ws/src/my_simulations/txts/'+file_name
	print(txt_path)
	raw_map = np.loadtxt(txt_path, dtype = 'int')
	return np.array(raw_map)
################################################################################################################
################################################################################################################

yaw_stamp = 0

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
velocity_msg = Twist()
velocity_msg.linear.x = 0  
velocity_msg.linear.y = 0  
velocity_msg.linear.z = 0  
velocity_msg.angular.x = 0  
velocity_msg.angular.y = 0  
velocity_msg.angular.z = 0 

current_distance = 0
speed = 0.2

section = {
    'front': 0,
    'left': 0,
    'right': 0,
}

'''
Function: callback: This function callback computes current state of the robot with euler transformation.
'''

def callback(msg):
	global curr_state, first_yaw, yaw_stamp, theta

	quaternion = (
	msg.pose.pose.orientation.x,
	msg.pose.pose.orientation.y,
	msg.pose.pose.orientation.z,
	msg.pose.pose.orientation.w)

	rot_q = msg.pose.pose.orientation
	(roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

	euler = euler_from_quaternion(quaternion)
	roll = euler[0]
	pitch = euler[1]
	yaw = euler[2]

	curr_state = [msg.pose.pose.position.x,msg.pose.pose.position.y,yaw, degrees(yaw)]
	if yaw_stamp == 0:
		first_yaw = curr_state[3]
		yaw_stamp = 1

'''
Function: laser callback: This function collects and saves data from the laser scan.
'''

def callback_laser(msg):
	global section, left, right, front, back

	laser_range = np.array(msg.ranges)
	left = laser_range[90]
	front = laser_range[0]
	right = laser_range[270]
	back = laser_range[180]
	section = {
        'front': min(min(laser_range[340:359]), min(laser_range[0:20])),
        'left': min(laser_range[70:110]),
        'right': min(laser_range[250:290]),
		'front_dir' : laser_range[0],
		'left_dir' : laser_range[270],
		'right_dir' : laser_range[90],
    }

'''
Function: go_to_goal: This function publishes velocity regarding to the direction and actual state 
for getting to the given goal.
'''

def go_to_goal(xx, yy):
	global path_done, path_index, stop_stamp, path, end
	rotate_speed = 0.5
	forward_speed = 0.15
	msg = Twist()
	msg.linear.x = 0.0
	msg.linear.y = 0.0
	msg.linear.z = 0.0
	msg.angular.x = 0.0
	msg.angular.y = 0.0
	msg.angular.z = 0.0     

	goal = Point()
	goal.x = xx
	goal.y = yy
	
	print("next node: "+str(xx), str(yy))

	stop_stamp = 0
	while stop_stamp == 0:
		inc_x = goal.x - curr_state[0]
		inc_y = goal.y - curr_state[1]
		angle_to_goal = atan2(inc_y, inc_x)
		two_point_distance_to_goal = np.sqrt(inc_x*inc_x + inc_y*inc_y)
		anglediff = fmod((angle_to_goal - theta), 2*pi)

		if (anglediff < 0.0):
			if (abs(anglediff) > (2*pi + anglediff)):
				anglediff = 2*pi + anglediff
		else:
			if (anglediff > abs(anglediff - 2*pi)):
				anglediff = anglediff - 2*pi

		if abs(anglediff) > 0.1:
			msg.angular.z = anglediff #dir * angSpeed
			msg.linear.x = 0
			pub.publish(msg)
			rospy.Rate(5).sleep()

		if(two_point_distance_to_goal >= 0.05):
			msg.linear.x = min(forward_speed, two_point_distance_to_goal)
			msg.angular.z = 0.0
			pub.publish(msg)
			rospy.Rate(5).sleep()
		else:
			stop_stamp = 1
			if(heading=='e'):
				while (abs(0 - theta) > 0.1):
					msg.linear.x = 0
					msg.angular.z = 0 - theta
					pub.publish(msg)
			elif(heading=='s'):
				while abs(-pi/2 + abs(theta)) > 0.1:
					msg.linear.x = 0
					msg.angular.z = -pi/2 + abs(theta)
					pub.publish(msg)
			elif(heading=='w'):
				while (pi - abs(theta) > 0.1):
					msg.linear.x = 0
					if theta > 0:
						msg.angular.z = pi - theta
					else:
						msg.angular.z = pi + theta
					pub.publish(msg)
			if(heading=='n'):
				while (abs(pi/2 - theta) > 0.1):
					msg.linear.x = 0
					msg.angular.z = pi/2 - theta
					pub.publish(msg)
				
	msg.linear.x = 0.0
	msg.angular.z = 0.0
	pub.publish(msg)		#stop the robot

	if round(curr_state[0]) == end[0] and round(curr_state[1]) == end[1]:
		path.append(end)
		path_done = 1
	else:
		tmp = []
		tmp = (round(curr_state[0]), round(curr_state[1]))
		path.append(tmp)

'''
Function: heading: This function calls function for the logic based on the robot states.
'''

def heading_east_movement():
	global heading, moved_ahead, curr_state, left, front, right, a, back, wall_found
	print("-------------------------------------------------")
	print('heading east')
	print("sensor left, front, right, back, moved_ahead, heading")
	print(left, front, right, back, moved_ahead, heading)
	print("current state: "+str(curr_state[0]), str(curr_state[1]))


	if left < a or front < a or right < a:
		wall_found = 1
	elif wall_found == 0:
		go_to_goal(round(curr_state[0] + 1), round(curr_state[1]))		#go ahead

	if wall_found == 1:
		if left > a and moved_ahead == 1:	#turn left and move on...outter corner
			heading = 'n'
			go_to_goal(round(curr_state[0]), round(curr_state[1] + 1))
			moved_ahead = 0
		elif left < a and front > a:		#go ahead
			go_to_goal(round(curr_state[0] + 1), round(curr_state[1]))
			moved_ahead = 1
		elif front < a and right > a:		#turn right
			heading = 's'
			go_to_goal(round(curr_state[0]), round(curr_state[1] - 1))
			moved_ahead = 1
		elif front < a and right < a and back < a:	#turn left
			heading = 'n'
			go_to_goal(curr_state[0], curr_state[1] + 1)
			moved_ahead = 1
		elif (front < a or front > a) and right < a:		#turn around
			heading = 'w'
			go_to_goal(round(curr_state[0] - 1), round(curr_state[1]))
			moved_ahead = 1
	
	

	'''
	if left > a and front > a and left > a:		#move ahead
		if moved_ahead == 1:					#turn left and move on
			go_to_goal(curr_state[0], curr_state[1] + 1)
			heading = 'n'
			moved_ahead = 0
		else:
			go_to_goal(curr_state[0] + 1, curr_state[1])
			moved_ahead = 0
	elif left > a and moved_ahead == 1:			#outter corner
		go_to_goal(curr_state[0], curr_state[1] + 1)
		heading = 'n'
		moved_ahead = 0
	elif left > a and front < a and right > a:	#turn right
		go_to_goal(curr_state[0], curr_state[1] - 1)
		moved_ahead = 1
		heading = 's'
	elif left < a and front < a and right > a:	#turn right
		go_to_goal(curr_state[0], curr_state[1] - 1)
		moved_ahead = 1
		heading = 's'
	elif left > a and front < a and right > a:	#turn around
		go_to_goal(curr_state[0] - 1, curr_state[1])
		moved_ahead = 1
		heading = 'w'
	elif left > a and front > a and right < a:	#turn around
		go_to_goal(curr_state[0] - 1, curr_state[1])
		moved_ahead = 1
		heading = 'w'
	elif front > a and left < a:				#move ahead
		go_to_goal(curr_state[0] + 1, curr_state[1])
		moved_ahead = 1
	'''

def heading_south_movement():
	global heading, moved_ahead, curr_state, left, front, right, back, a, wall_found
	print("-------------------------------------------------")
	print('heading south')
	print("sensor left, front, right, back, moved_ahead, heading")
	print(left, front, right, back, moved_ahead, heading)
	print("current state: "+str(curr_state[0]), str(curr_state[1]))

	if left < a or front < a or right < a:
		wall_found == 1
	elif wall_found == 0:
		go_to_goal(round(curr_state[0]), round(curr_state[1] - 1))		#go ahead

	if wall_found == 1:
		if left > a and moved_ahead == 1:	#turn left and move on...outter corner
			heading = 'e'
			go_to_goal(round(curr_state[0]) + 1, round(curr_state[1]))
			moved_ahead = 0
		elif left < a and front > a:		#go ahead
			go_to_goal(round(curr_state[0]), round(curr_state[1] - 1))
			moved_ahead = 1
		elif front < a and right > a:		#turn right
			heading = 'w'
			go_to_goal(round(curr_state[0] - 1), round(curr_state[1]))
			moved_ahead = 1
		elif front < a and right < a and back < a:	#turn left
			heading = 'e'
			go_to_goal(round(curr_state[0] + 1), round(curr_state[1]))
			moved_ahead = 1
		elif (front < a or front > a) and right < a:		#turn around
			heading = 'n'
			go_to_goal(round(curr_state[0]), round(curr_state[1] + 1))
			moved_ahead = 1
	


	

	'''
	if left > a and front > a and left > a:		#move ahead
		if moved_ahead == 1:					#turn left and move on
			go_to_goal(curr_state[0] + 1, curr_state[1])
			heading = 'e'
			moved_ahead = 0
		else:
			go_to_goal(curr_state[0] - 1, curr_state[1])
			moved_ahead = 0
	elif left > a and moved_ahead == 1:
			go_to_goal(curr_state[0] + 1, curr_state[1])
			heading = 'e'
			moved_ahead = 0	
	elif left > a and front < a and right > a:	#turn right
		go_to_goal(curr_state[0] - 1, curr_state[1])
		moved_ahead = 1
		heading = 'w'
	elif left < a and front < a and right > a:	#turn right
		go_to_goal(curr_state[0] - 1, curr_state[1])
		moved_ahead = 1
		heading = 'w'
	elif left > a and front < a and right > a:	#turn around
		go_to_goal(curr_state[0], curr_state[1] + 1)
		moved_ahead = 1
		heading = 'n'
	elif left > a and front > a and right < a:	#turn around
		go_to_goal(curr_state[0], curr_state[1] + 1)
		moved_ahead = 1
		heading = 'n'
	elif front > a and left < a:				#move ahead
		go_to_goal(curr_state[0], curr_state[1] - 1)
		moved_ahead = 1

	elif front > a:				#move ahead
		go_to_goal(curr_state[0], curr_state[1] - 1)
		moved_ahead = 1
	elif front < a and right < a:	#turn around
		go_to_goal(curr_state[0], curr_state[1] + 1)
		moved_ahead = 1
		heading = 'n'
	elif front < a and right > a:	#turn right
		go_to_goal(curr_state[0] - 1, curr_state[1])
		moved_ahead = 1
		heading = 'w'
	'''

def heading_west_movement():
	global heading, moved_ahead, curr_state, left, front, right, a, back, wall_found
	print("-------------------------------------------------")
	print('heading west')
	print("sensor left, front, right, back, moved_ahead, heading")
	print(left, front, right, back, moved_ahead, heading)
	print("current state: "+str(curr_state[0]), str(curr_state[1]))
	if left < a or front < a or right < a:
		wall_found == 1
	elif wall_found == 0:
		go_to_goal(round(curr_state[0] - 1), round(curr_state[1]))		#go ahead
	
	if wall_found == 1:
		if left > a and moved_ahead == 1:	#turn left and move on...outter corner
			heading = 's'
			go_to_goal(round(curr_state[0]), round(curr_state[1] - 1))
			moved_ahead = 0
		elif left < a and front > a:		#go ahead
			go_to_goal(round(curr_state[0] - 1), round(curr_state[1]))
			moved_ahead = 1
		elif front < a and right > a:		#turn right
			heading = 'n'
			go_to_goal(round(curr_state[0]), round(curr_state[1] + 1))
			moved_ahead = 1
		elif front < a and right < a and back < a:	#turn left
			heading = 's'
			go_to_goal(round(curr_state[0]), round(curr_state[1] - 1))
			moved_ahead = 1
		elif (front < a or front > a) and right < a:		#turn around
			heading = 'e'
			go_to_goal(round(curr_state[0] + 1), round(curr_state[1]))
			moved_ahead = 1

	
	
	'''
	if left > a and front > a and left > a:		#move ahead
		if moved_ahead == 1:					#turn left and move on
			go_to_goal(curr_state[0], curr_state[1] - 1)
			heading = 's'
			moved_ahead = 0
		else:
			go_to_goal(curr_state[0] - 1, curr_state[1])
			moved_ahead = 0
			heading = 'w'
	elif left > a and moved_ahead ==1:			#outter corner
		go_to_goal(curr_state[0], curr_state[1] - 1)
		heading = 's'
		moved_ahead = 0
	elif left > a and front < a and right > a:	#turn right
		go_to_goal(curr_state[0], curr_state[1] + 1)
		moved_ahead = 1
		heading = 'n'
	elif left < a and front < a and right > a:	#turn right
		go_to_goal(curr_state[0], curr_state[1] + 1)
		moved_ahead = 1
		heading = 'n'
	elif left > a and front < a and right > a:	#turn around
		go_to_goal(curr_state[0] + 1, curr_state[1])
		moved_ahead = 1
		heading = 'e'
	elif left > a and front > a and right < a:	#turn around
		go_to_goal(curr_state[0] + 1, curr_state[1])
		moved_ahead = 1
		heading = 'e'
	elif front > a and left < a:				#move ahead
		go_to_goal(curr_state[0] - 1, curr_state[1])
		moved_ahead = 1
	'''

def heading_north_movement():
	global heading, moved_ahead, curr_state, left, front, right, a, back, wall_found
	print("-------------------------------------------------")
	print('heading north')
	print("sensor left, front, right, back, moved_ahead, heading")
	print(left, front, right, back, moved_ahead, heading)
	print("current state: "+str(curr_state[0]), str(curr_state[1]))

	if left < a or front < a or right < a:
		wall_found == 1
	elif wall_found == 0:
		go_to_goal(round(curr_state[0]), round(curr_state[1] + 1))		#go ahead
	
	if wall_found == 1:
		if left > a and moved_ahead == 1:	#turn left and move on...outter corner
			heading = 'w'
			go_to_goal(round(curr_state[0] - 1), round(curr_state[1]))
			moved_ahead = 0	
		elif left < a and front > a:		#go ahead
			go_to_goal(round(curr_state[0]), round(curr_state[1] + 1))
			moved_ahead = 1
		elif front < a and right > a:		#turn right
			heading = 'e'
			go_to_goal(round(curr_state[0] + 1), round(curr_state[1]))
			moved_ahead = 1
		elif front < a and right < a and back < a:	#turn left
			heading = 'w'
			go_to_goal(round(curr_state[0] - 1), round(curr_state[1]))
			moved_ahead = 1
		elif (front < a or front > a) and right < a:		#turn around
			heading = 's'
			go_to_goal(round(curr_state[0]), round(curr_state[1] - 1))
			moved_ahead = 1
			




	'''
	if left > a and front > a and left > a:		#move ahead
		if moved_ahead == 1:					#turn left and move on
			go_to_goal(curr_state[0] - 1, curr_state[1])
			heading = 'w'
			moved_ahead = 0
		else:
			go_to_goal(curr_state[0], curr_state[1] + 1)
			moved_ahead = 0
	elif left > a and moved_ahead == 1:			#outter corner
		go_to_goal(curr_state[0] - 1, curr_state[1])
		heading = 'w'
		moved_ahead = 0
	elif front < a and right > a:	#turn right
		go_to_goal(curr_state[0] + 1, curr_state[1])
		moved_ahead = 1
		heading = 'e'
	elif left > a and front < a and right > a:	#turn around
		go_to_goal(curr_state[0], curr_state[1] - 1)
		moved_ahead = 1
		heading = 's'
	elif left > a and front > a and right < a:	#turn around
		go_to_goal(curr_state[0], curr_state[1] - 1)
		moved_ahead = 1
		heading = 's'
	elif front > a and left < a:				#move ahead
		go_to_goal(curr_state[0], curr_state[1] + 1)
		moved_ahead = 1
	'''

rospy.Subscriber('/odom', Odometry, callback)			#odometry subscriber
rospy.Subscriber('/scan', LaserScan, callback_laser)	#laser scan subscriber

def main():
	global pub, right, left, front, back, curr_state, moved_ahead, a, heading, path, path_done, end, wall_found		#global variables

	'''
        HANDLING THE ARGUMENTS
    '''
	ap = argparse.ArgumentParser()
	ap.add_argument("-hd", "--heading", required=False, type=str)
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
		end = (args.final_x, args.final_y)
	else:
		sys.exit("Wrong final node position entered")

	if args.heading == "NULL":
		heading = 'e'
		print("No heading value was entered. Remember, for entering the north heading value type: -hd 'n'. 'e' for east and so on. Default heading is east.")
	else: 
		heading = args.heading
	###########################################################
    ###########################################################

	rospy.init_node('wall_follower')	#initialize the node named wall_follower for the launch file
	rospy.sleep(2)

	a = 0.75  # maximum threshold distance

	moved_ahead = 0
	right = 0
	front = 0
	left = 0
	wall_found = 0

	path_done = 0		#stamp for path done...0 = not in the goal...1 = in the goal
	start = (round(curr_state[0]), round(curr_state[1]))
	path = []
	path.append(start)

	
	t0 = rospy.Time.now().to_sec()		#handling the time start

	'''
		GRAPH PLOTTING
	'''
	plt.figure(figsize=(13, 13), dpi=80)
	plt.imshow(maze[::-1], cmap='binary', interpolation = 'nearest', origin='lower')
	plt.title("Turtlebot finding the optimal path by Wall follower algorithm in the given map", pad = 20, fontsize='large', fontweight='bold')
	plt.xlabel("X axis of the map", fontsize = 'large')
	plt.ylabel("Y axis of the map", fontsize = 'large')

	listOf_Xticks = np.arange(0, len(maze), 1)
	plt.xticks(listOf_Xticks)       #setting the interval of x-axis
	listOf_Yticks = np.arange(0, len(maze[0]), 1)
	plt.yticks(listOf_Yticks)       #setting the interval of y-axis

	#plt.grid()
	plt.plot([i[0] for i in path], [i[1] for i in path], label="optimal path")
	plt.plot(start[0], start[1], 'ro', markersize=15, label="start node")
	plt.plot(end[0], end[1], 'bo', markersize=15, label="end node")
	plt.legend()
	#######################################################################
	#######################################################################

	while not rospy.is_shutdown():
		if path_done == 0:
			if heading == 'e':
				heading_east_movement()

			elif heading == 's':
				heading_south_movement()

			elif heading == 'w':
				heading_west_movement()

			elif heading == 'n':
				heading_north_movement()
		else:
			print("Robot came to the final node")
			t_all = rospy.Time.now().to_sec() - t0
			print("Overall time for robot to find the optimal path and get to the final node: "+str(round(t_all, 4)))
			print("Optimal path length                                                      : "+str(len(path) - 1))
			print(path)
			plt.show()

	rospy.spin()


if __name__ == '__main__':
    main()