import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """
        # Define sampling time
        self.dt = dt
        # Define the  control input variables
        self.u = np.block([[u_x],[u_y]])
        # Initial State
        # Define the State Transition Matrix A
        self.A = np.block([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # Define the Control Input Matrix B
        self.B = np.block([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])
        # Define Measurement Mapping Matrix
        self.H = np.block([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        #Initial Process Noise Covariance
        self.Q = np.block([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        # Initial Measurement Noise Covariance
        self.R = np.block([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
        # Initial Covariance Matrix

    def predict(self, x, P):
        # x_k =Ax_(k-1) + Bu_(k-1)
        # a priori estimate
        x = np.dot(self.A, x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q
        P = np.dot(np.dot(self.A, P), self.A.T) + self.Q
        return x, P



    def update(self, z, x, P):
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(S)
        K = np.dot(np.dot(P, self.H.T), np.linalg.inv(S))
        x = np.round(x + np.dot(K, (z - np.dot(self.H, x))))
        I = np.eye(self.H.shape[1])
        P = np.dot((I - np.dot(K, self.H)), P)
        return x, P


