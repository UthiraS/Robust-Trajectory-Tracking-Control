#!/usr/bin/env python3
from math import pi, sqrt, atan2, cos, sin, copysign, asin
from turtle import position
import numpy as np
from numpy import NaN
import rospy
import tf
from std_msgs.msg import Empty, Float32
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Twist, Pose2D
import pickle
import os
import matplotlib.pyplot as plt

class Quadrotor():
    def __init__(self):
        # publisher for rotor speeds
        self.motor_speed_pub = rospy.Publisher("/crazyflie2/command/motor_speed", Actuators, queue_size=10)
        # subscribe to Odometry topic
        self.odom_sub = rospy.Subscriber("/crazyflie2/ground_truth/odometry", Odometry, self.odom_callback)
        self.t0 = None
        self.t = None
        self.t_series = []
        self.x_series = []
        self.y_series = []
        self.z_series = []
        self.mutex_lock_on = False
        rospy.on_shutdown(self.save_data)
        # TODO: include initialization codes if needed
        self.g = 9.81
        self.m = 0.027
        self.l = 46e-3
        
        self.Ix = 16.571710*1e-6
        self.Iy = 16.571710*1e-6
        self.Iz = 29.261652*1e-6
        self.Ip = 12.65625*1e-8
        
        self.kF = float(1.28192e-8)
        self.kM = float(5.964552e-3)
        
        self.w_max = 2618
        self.w_min = 0
        
        self.lamda_phi = 12
        self.lamda_z = 5 
        self.lamda_theta = 12
        self.lamda_psi = 5

        self.K0_theta = 135
        self.K0_z = 10
        self.K0_phi = 140
        self.K0_psi = 25

        self.phi = 1
        
        self.kp = 120
        self.kd = 8

        self.xd = []
        self.yd = []
        self.zd = []
        self.xd_dot = []
        self.yd_dot = []
        self.zd_dot = []
        self.xd_ddot = []
        self.yd_ddot = []
        self.zd_ddot = []
        self.T = []
        
    def traj_evaluate(self): 
        # TODO: evaluating the corresponding trajectories designed in Part 1 to return the desired positions, velocities and accelerations
        
            q_d = np.array([[0, 0, (6*self.t**5)/3125 - (3*self.t**4)/125 + (2*self.t**3)/25 + (1351079888211149*self.t**2)/81129638414606681695789005144064], [(2332032831046741*self.t**5)/295147905179352825856 - self.t**4/2025 + (22*self.t**3)/2025 - (8*self.t**2)/81 + (32*self.t)/81 - 47/81, 0, 1], [1, (1166016415523379*self.t**5)/147573952589676412928 - (11*self.t**4)/10125 + (118*self.t**3)/2025 - (616*self.t**2)/405 + (1568*self.t)/81 - 7808/81, 1], [- (2332032831045997*self.t**5)/295147905179352825856 + (1935769439833103*self.t**4)/1152921504606846976 - (1272127894743261*self.t**3)/9007199254740992 + (1654099863138637*self.t**2)/281474976710656 - (266054665486453*self.t)/2199023255552 + 4343749640595151/4398046511104, 1, 1], [0, - (145752051940345*self.t**5)/18446744073709551616 + (1309491091651539*self.t**4)/576460752303423488 - (73113993950483*self.t**3)/281474976710656 + (4156099656120617*self.t**2)/281474976710656 - (3670468446302185*self.t)/8796093022208 + 1289825552458379/274877906944, 1]])
            q_d_dot = np.array([[0, 0, (6*self.t**4)/625 - (12*self.t**3)/125 + (6*self.t**2)/25 + (1351079888211149*self.t)/40564819207303340847894502572032], [(11660164155233705*self.t**4)/295147905179352825856 - (4*self.t**3)/2025 + (22*self.t**2)/675 - (16*self.t)/81 + 32/81, 0, 0], [0, (5830082077616895*self.t**4)/147573952589676412928 - (44*self.t**3)/10125 + (118*self.t**2)/675 - (1232*self.t)/405 + 1568/81, 0], [-(11660164155229985*self.t**4)/295147905179352825856 + (1935769439833103*self.t**3)/288230376151711744 - (3816383684229783*self.t**2)/9007199254740992 + (1654099863138637*self.t)/140737488355328 - 266054665486453/2199023255552, 0, 0], [0, - (728760259701725*self.t**4)/18446744073709551616 + (1309491091651539*self.t**3)/144115188075855872 - (219341981851449*self.t**2)/281474976710656 + (4156099656120617*self.t)/140737488355328 - 3670468446302185/8796093022208, 0]])
            q_d_ddot = np.array([[0, 0, (24*self.t**3)/625 - (36*self.t**2)/125 + (12*self.t)/25 + 1351079888211149/40564819207303340847894502572032], [(11660164155233705*self.t**3)/73786976294838206464 - (4*self.t**2)/675 + (44*self.t)/675 - 16/81, 0, 0], [0, (5830082077616895*self.t**3)/36893488147419103232 - (44*self.t**2)/3375 + (236*self.t)/675 - 1232/405, 0], [-(11660164155229985*self.t**3)/73786976294838206464 + (5807308319499309*self.t**2)/288230376151711744 - (3816383684229783*self.t)/4503599627370496 + 1654099863138637/140737488355328, 0, 0], [0, - (728760259701725*self.t**3)/4611686018427387904 + (3928473274954617*self.t**2)/144115188075855872 - (219341981851449*self.t)/140737488355328 + 4156099656120617/140737488355328, 0]])

            if self.t <= 5:
                x_d = q_d[0,0]
                x_d_dot = q_d_dot[0,0]
                x_d_ddot = q_d_ddot[0,0]
                y_d = q_d[0,1]
                y_d_dot = q_d_dot[0,1]
                y_d_ddot = q_d_ddot[0,1]
                z_d = q_d[0,2]
                z_d_dot = q_d_dot[0,2] 
                z_d_ddot = q_d_ddot[0,2]


            elif self.t>5 and self.t<=20:
                x_d = q_d[1,0]
                x_d_dot = q_d_dot[1,0]
                x_d_ddot = q_d_ddot[1,0]
                y_d = q_d[1,1]
                y_d_dot = q_d_dot[1,1]
                y_d_ddot = q_d_ddot[1,1]
                z_d = q_d[1,2]
                z_d_dot = q_d_dot[1,2]
                z_d_ddot = q_d_ddot[1,2]


            elif self.t>20 and self.t<=35:
                x_d = q_d[2,0]
                x_d_dot = q_d_dot[2,0]
                x_d_ddot = q_d_ddot[2,0]
                y_d = q_d[2,1]
                y_d_dot = q_d_dot[2,1]
                y_d_ddot = q_d_ddot[2,1]
                z_d = q_d[2,2]
                z_d_dot = q_d_dot[2,2]
                z_d_ddot = q_d_ddot[2,2]


            elif self.t>35 and self.t<=50:
                x_d = q_d[3,0]
                x_d_dot = q_d_dot[3,0]
                x_d_ddot = q_d_ddot[3,0]
                y_d = q_d[3,1]
                y_d_dot = q_d_dot[3,1]
                y_d_ddot = q_d_ddot[3,1]
                z_d = q_d[3,2]
                z_d_dot = q_d_dot[3,2]
                z_d_ddot = q_d_ddot[3,2]

            elif self.t>50 and self.t<=65:
                x_d = q_d[4,0]
                x_d_dot = q_d_dot[4,0]
                x_d_ddot = q_d_ddot[4,0]
                y_d = q_d[4,1]
                y_d_dot = q_d_dot[4,1]
                y_d_ddot = q_d_ddot[4,1]
                z_d = q_d[4,2]
                z_d_dot = q_d_dot[4,2]
                z_d_ddot = q_d_ddot[4,2]
                
            
            elif self.t>65:
                x_d = 0
                x_d_dot = 0
                x_d_ddot = 0
                y_d = 0
                y_d_dot = 0
                y_d_ddot = 0
                z_d = 1
                z_d_dot = 0
                z_d_ddot = 0

            #Uncomment for Plots
            # if self.t == 65:
            #    figure, axis = plt.subplots(3, 3)
            #    axis[0, 0].plot(self.T, self.xd)
            #    axis[0, 0].set_title("Xdesired Trajectory")
            #    axis[0, 1].plot(self.T, self.yd)
            #    axis[0, 1].set_title("Ydesired Trajectory")
            #    axis[0, 2].plot(self.T, self.zd)
            #    axis[0, 2].set_title("Zdesired Trajectory")
            #    axis[1, 0].plot(self.T, self.xd_dot)
            #    axis[1, 0].set_title("X_dot_desired Trajectory")
            #    axis[1, 1].plot(self.T, self.yd_dot)
            #    axis[1, 1].set_title("Y_dot_desired Trajectory")
            #    axis[1, 2].plot(self.T, self.zd_dot)
            #    axis[1, 2].set_title("Z_dot_desired Trajectory")
            #    axis[2, 0].plot(self.T, self.xd_ddot)
            #    axis[2, 0].set_title("X_ddot_desired Trajectory")
            #    axis[2, 1].plot(self.T, self.yd_ddot)
            #    axis[2, 1].set_title("Y_ddot_desired Trajectory")
            #    axis[2, 2].plot(self.T, self.zd_ddot)
            #    axis[2, 2].set_title("Z_ddot_desired Trajectory")
            #    plt.show()
               
  
            # self.xd.append(x_d)
            # self.yd.append(y_d)
            # self.zd.append(z_d)
            # self.xd_dot.append(x_d_dot)
            # self.yd_dot.append(y_d_dot)
            # self.zd_dot.append(z_d_dot)
            # self.xd_ddot.append(x_d_ddot)
            # self.yd_ddot.append(y_d_ddot)
            # self.zd_ddot.append(z_d_ddot)
            # self.T.append(self.t)
            
            
            return [x_d, y_d, z_d, x_d_dot, y_d_dot, z_d_dot, x_d_ddot, y_d_ddot, z_d_ddot]

            

    def smc_control(self, xyz, xyz_dot, rpy, rpy_dot):
        
        # obtain the desired values by evaluating the corresponding trajectories
        [x_d, y_d, z_d, x_d_dot, y_d_dot, z_d_dot, x_d_ddot, y_d_ddot, z_d_ddot] = self.traj_evaluate()
        
        # TODO: implement the Sliding Mode Control laws designed in Part 2 to calculate the control inputs "u"
        # REMARK: wrap the roll-pitch-yaw angle errors to [-pi to pi]
    
        omega = 0
        xyz_d = np.array([[x_d], [y_d], [z_d]])
        xyz_d_dot = np.array([[x_d_dot], [y_d_dot], [z_d_dot]])
        
        e_xyz = xyz - xyz_d
        e_xyz_dot = xyz_dot - xyz_d_dot

        s_xyz = e_xyz_dot + (self.lamda_z*e_xyz)
        sz = s_xyz[2, 0]

        satz = min(max(sz/self.phi, -1), 1)

        # ur1 = -self.K0_z * satz
        u1 = (self.m*(self.g + z_d_ddot - self.lamda_z*(xyz_dot[2, 0] - z_d_dot)))/(cos(rpy[0, 0])*cos(rpy[1, 0])) + ((-self.K0_z * satz) * self.m /(cos(rpy[0, 0])*cos(rpy[1, 0])))

        Fx = self.m*(-self.kp*(xyz[0, 0]-x_d) - self.kd*(xyz_dot[0, 0]-x_d_dot) + x_d_ddot)
        Fy = self.m*(-self.kp*(xyz[1, 0]-y_d) - self.kd*(xyz_dot[1, 0]-y_d_dot) + y_d_ddot)
        
        theta_d = asin(Fx/u1)
        phi_d = asin(-Fy/u1)
        psi_d = 0
        phid_dot = 0
        thetad_dot = 0
        psid_dot = 0

        phid_ddot = 0
        thetad_ddot = 0
        psid_ddot = 0

        q_d_rpy = np.array([[phi_d], [theta_d], [psi_d]])
        q_d_dot_rpy = np.array([[phid_dot], [thetad_dot], [psid_dot]])
        
        e_rpy = rpy - q_d_rpy

        for i, a in enumerate(e_rpy):
            while a < pi:
                a += 2 * pi
            while a > pi:
                a -= 2 * pi
            e_rpy[i,0] = a


        e_dot_rpy = rpy_dot - q_d_dot_rpy
        
        s2 = e_dot_rpy[0,0] + (self.lamda_phi*e_rpy[0,0])
        s3 = e_dot_rpy[1,0] + (self.lamda_theta*e_rpy[1,0])
        s4 = e_dot_rpy[2,0] + (self.lamda_psi*e_rpy[2,0])
        
        sat2 = min(max(s2/self.phi, -1), 1)
        sat3 = min(max(s3/self.phi, -1), 1)
        sat4 = min(max(s4/self.phi, -1), 1)

        
        u2 = self.Ix*(phid_ddot + self.lamda_phi*(phid_dot - rpy_dot[0, 0]) + (self.Ip*omega*rpy_dot[1, 0])/self.Ix - (rpy_dot[1, 0]*rpy_dot[2, 0]*(self.Iy - self.Iz))/self.Ix) + (-self.K0_phi * sat2)*self.Ix
        u3 = self.Iy*(thetad_ddot - self.lamda_theta*(rpy_dot[1, 0] - thetad_dot) - (self.Ip*omega*rpy_dot[0, 0])/self.Iy + (rpy_dot[0, 0]*rpy_dot[2, 0]*(self.Ix - self.Iz))/self.Iy) + (-self.K0_theta * sat3)*self.Iy
        u4 = self.Iz*(psid_ddot + self.lamda_psi*(psid_dot - rpy_dot[2, 0]) - (rpy_dot[0, 0]*rpy_dot[1, 0]*(self.Ix - self.Iy))/self.Iz) + (-self.K0_psi * sat4)*self.Iz

        u = np.array([[u1], [u2], [u3], [u4]])
        
        # TODO: convert the desired control inputs "u" to desired rotor velocities "motor_vel" by using the "allocation matrix"
        k_s = np.array([[1/(4*self.kF), -sqrt(2)/(4*self.kF*self.l), -sqrt(2)/(4*self.kF*self.l), -1/(4*self.kM*self.kF)], [1/(4*self.kF), -sqrt(2)/(4*self.kF*self.l), sqrt(2)/(4*self.kF*self.l), 1/(4*self.kM*self.kF)], [1/(4*self.kF), sqrt(2)/(4*self.kF*self.l), sqrt(2)/(4*self.kF*self.l), -1/(4*self.kM*self.kF)], [1/(4*self.kF), sqrt(2)/(4*self.kF*self.l), -sqrt(2)/(4*self.kF*self.l), 1/(4*self.kM*self.kF)]])
        
        w_mat = np.matmul(k_s,u)
        
        w1 = sqrt(w_mat[0])
        w2 = sqrt(w_mat[1])
        w3 = sqrt(w_mat[2])
        w4 = sqrt(w_mat[3])
       
        print("omega = ", omega)
        # TODO: maintain the rotor velocities within the valid range of [0 to 2618]
        if w1>=2618:
            w1=2618
        else:
            print("in range")
        if w2>=2618:
            w2=2618
        else:
            print("in range")
        if w3>=2618:
            w3=2618
        else:
            print("in range")
        if w4>=2618:
            w4=2618
        else:
            print("in range")
        motor_vel = np.array([[w1], [w2], [w3], [w4]])
        omega = w1-w2+w3-w4
        # publish the motor velocities to the associated ROS topic
        motor_speed = Actuators()
        motor_speed.angular_velocities = [motor_vel[0,0], motor_vel[1,0], motor_vel[2,0], motor_vel[3,0]]
        self.motor_speed_pub.publish(motor_speed)
        return xyz, xyz_dot, rpy, rpy_dot

    # odometry callback function (DO NOT MODIFY)
    def odom_callback(self, msg):
        # print('3')
        if self.t0 == None:
            self.t0 = msg.header.stamp.to_sec()
        self.t = msg.header.stamp.to_sec() - self.t0

        # convert odometry data to xyz, xyz_dot, rpy, and rpy_dot
        w_b = np.asarray([[msg.twist.twist.angular.x], [msg.twist.twist.angular.y], [msg.twist.twist.angular.z]])
        v_b = np.asarray([[msg.twist.twist.linear.x], [msg.twist.twist.linear.y], [msg.twist.twist.linear.z]])
        xyz = np.asarray([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.pose.pose.position.z]])
        q = msg.pose.pose.orientation
        T = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0:3, 3] = xyz[0:3, 0]
        R = T[0:3, 0:3]
        xyz_dot = np.dot(R, v_b)
        rpy = tf.transformations.euler_from_matrix(R, 'sxyz')
        rpy_dot = np.dot(np.asarray([[1, np.sin(rpy[0])*np.tan(rpy[1]), np.cos(rpy[0])*np.tan(rpy[1])], [0, np.cos(rpy[0]), -np.sin(rpy[0])], [0, np.sin(rpy[0])/np.cos(rpy[1]), np.cos(rpy[0])/np.cos(rpy[1])]]), w_b)
        rpy = np.expand_dims(rpy, axis=1)

        # store the actual trajectory to be visualized later
        if (self.mutex_lock_on is not True):
            self.t_series.append(self.t)
            self.x_series.append(xyz[0, 0])
            self.y_series.append(xyz[1, 0])
            self.z_series.append(xyz[2, 0])
        
        # call the controller with the current states
        self.smc_control(xyz, xyz_dot, rpy, rpy_dot)

        # save the actual trajectory data 
    def save_data(self):
        # TODO: update the path below with the correct path
        with open("/home/chinmayee/rbe502_project/src/project/drone/log.pkl", "wb") as fp:
            self.mutex_lock_on = True
            pickle.dump([self.t_series,self.x_series,self.y_series,self.z_series], fp)

if __name__ == '__main__':
    rospy.init_node("quadrotor_control")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = Quadrotor()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

       
       
       

