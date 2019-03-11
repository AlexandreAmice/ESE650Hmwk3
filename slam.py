from __future__ import division
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2
from math import cos, sin, tan, atan2, acos, asin
from sys import maxsize as MAXINT
from prettytable import PrettyTable

def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi).
        UPDATE: I will return a 3+2m x 3+ 2m matrix instead
    '''

    ###
    # Implement the vehicle model and its Jacobian you derived.
    ###
    xv1 = ekf_state['x'][0]
    yv1 = ekf_state['x'][1]
    phi1 = ekf_state['x'][2]
    a = vehicle_params['a']
    b = vehicle_params['b']
    L = vehicle_params['L']
    H = vehicle_params['H']
    ve = u[0]
    alpha = u[1]
    vc = ve/(1-tan(alpha)*H/L)
    xv2 = xv1+dt*(vc*cos(phi1) - vc/L*tan(alpha)*(a*sin(phi1)+b*cos(phi1)))
    yv2 = yv1 + dt*(vc*sin(phi1) + vc/L*tan(alpha)*(a*cos(phi1)-b*sin(phi1)))
    phi2 = slam_utils.clamp_angle(phi1 + dt*vc/L*tan(alpha))
    # motion = np.array([xv2 - xv1, yv2 - yv1, phi2 - phi1])
    # motion[2] = slam_utils.clamp_angle(motion[2])
    dx = dt*(vc*cos(phi1) - vc/L*tan(alpha)*(a*sin(phi1)+b*cos(phi1)))
    dy = dt*(vc*sin(phi1) + vc/L*tan(alpha)*(a*cos(phi1)-b*sin(phi1)))
    dphi = slam_utils.clamp_angle(dt*vc/L*tan(alpha))
    motion = np.array([dx,dy,dphi])

    # G = np.array([[vc*cos(phi1)-a*sin(phi1)*vc*tan(alpha)-b*cos(phi1)*vc*tan(alpha), 0, dt*vc*tan(alpha)*(-vc*sin(phi1)-vc/L*tan(alpha)*(a*cos(phi1)-b*sin(phi1)))],
    #              [0, vc*sin(phi1)+a*cos(phi1)*vc*tan(alpha)-b*sin(phi1)*vc*tan(alpha), vc*tan(alpha)*(a*cos(phi1)-b*sin(phi1) + dt*(vc*cos(phi1)+vc/L*tan(alpha)*(-a*sin(phi1)-b*cos(phi1))))],
    #              [0, 0, vc*tan(alpha)]])
    Gx  = np.array([[1, 0, dt*(-vc*sin(phi2) - vc/L*tan(alpha)*(a*cos(phi2)-b*cos(phi2)))],
                   [0,1,dt*(vc*cos(phi2) + vc/L*tan(alpha)*(-a*sin(phi2)-b*cos(phi2)))],
                   [0,0,1]])

    G = spl.block_diag(Gx, np.eye(2*ekf_state["num_landmarks"]))


    return motion, G


def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''
    motion, G = motion_model(u,dt,ekf_state,vehicle_params)
    Fx = np.zeros((3, 3+2*ekf_state['num_landmarks']))
    Fx[0:3,0:3] = np.eye(3)
    xhat = ekf_state['x'][0:3]+motion
    xhat[2] = slam_utils.clamp_angle(xhat[2])
    R = np.diag([sigmas['xy'], sigmas['xy'], sigmas['phi']])
    # Sigmat = slam_utils.make_symmetric(G @ ekf_state['P'][0:3, 0:3] @ G.transpose() + R)
    Sigmat = slam_utils.make_symmetric(G @ ekf_state['P'] @ G.transpose() + Fx.transpose() @ R @ Fx)
    ekf_state['x'][0:3] = xhat
    ekf_state['P'] = Sigmat
    return ekf_state



def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    
    ###
    # Implement the GPS update.
    ###
    zhat = gps
    hxt = ekf_state['x'][0:2]
    Sigmat = ekf_state['P'][0:2,0:2]
    Q = sigmas['gps']*np.eye(2)
    r = zhat - hxt
    Ht = np.eye(2)
    Kt = Sigmat@Ht.transpose()@npl.inv(Ht@Sigmat@Ht.transpose() + Q.transpose())
    Sigmat = slam_utils.make_symmetric((np.eye(2) - Kt@Ht)@Sigmat)
    if r.transpose()@npl.inv(Q)@r > 13.8:
        return ekf_state
    ekf_state['x'][0:2] = ekf_state['x'][0:2]+Kt@r
    ekf_state['P'][0:2, 0:2] = Sigmat
    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
        UPDATE we return H as a 5 x 3+2*m matrix
    '''
    x = ekf_state['x'][0]
    y = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    xL = ekf_state['x'][2+(2*landmark_id-1)]
    yL = ekf_state['x'][3+(2*landmark_id-1)]
    dx = xL - x
    dy = yL - y
    dxdy = dx ** 2 + dy ** 2
    zhat = np.array([np.sqrt(dxdy), atan2(dy,dx)-phi+np.pi/2])
    Hupdate = np.array([[-dx*np.sqrt((dxdy)), -dy*np.sqrt((dxdy)), 0, dx*np.sqrt((dxdy)), dy*np.sqrt((dxdy))],
                        [dy/(dxdy), dx/(dxdy), -1, -dy/(dxdy), -dx/(dxdy)]])
    Fxj = np.zeros((5, ekf_state['P'].shape[0]))
    Fxj[0:3,0:3] = np.eye(3)
    Fxj[3,3+2*landmark_id-2] = 1
    Fxj[4, 3 + 2 * landmark_id-1] = 1
    H = Hupdate@Fxj
    return zhat, H


def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''
    (range, angle, diameter) = tree
    #TODO account for the diameter
    (r,b) = (range, angle)
    x, y, phi = ekf_state['x'][0],ekf_state['x'][1],ekf_state['x'][2]
    xL = x+r*cos(b+phi-np.pi/2)
    yL = y+r*sin(b+phi-np.pi/2)
    ekf_state['x'] = np.append(ekf_state['x'], (xL, yL))
    ekf_state['P'] = spl.block_diag(ekf_state['P'], np.eye(2))
    ekf_state['num_landmarks'] = ekf_state['num_landmarks']+1
    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    M = np.zeros((len(measurements), ekf_state["num_landmarks"]))
    Sigma = ekf_state['P']

    #Establish Q
    # Q = np.zeros_like(Sigma)
    # Q[0,0], Q[1,1], Q[2,2] = sigmas['xy'], sigmas['xy'], sigmas['phi']
    # for i in range(3,3+ekf_state['num_landmarks']):
    #     if i%2 == 1:
    #         Q[i] = sigmas['range']
    #     else:
    #         Q[i] = sigmas['bearing']
    Q = np.diag([sigmas['range'], sigmas['bearing']])

    #build cost
    for i,m in enumerate(measurements):
        (r, b, d) = m
        # TODO account for the diameter
        (r, b) = (r, b)
        for j in range(0,ekf_state['num_landmarks']):
            (zr, zb), H = laser_measurement_model(ekf_state, j+1)
            innov = np.array([r-zr, slam_utils.clamp_angle(b-zb)])
            residCov = H @ Sigma @ H.transpose() + Q.transpose()
            # Kt = Sigma @ H.transpose() @ npl.inv(H @ Sigma @ H.transpose() + Q.transpose())
            # SigmaT = (np.eye(Kt.shape[0]) - Kt @ H)@ Sigma
            M[i,j] = innov.transpose() @ npl.inv(residCov) @ innov
    matchings = slam_utils.solve_cost_matrix_heuristic(M.copy())

    assoc = np.zeros(len(measurements))
    for (i,j) in matchings:
        if M[i,j] > chi2.ppf(.9,2):
            assoc[i] = -2
        elif M[i,j] > chi2.ppf(0.5,2):
            assoc[i] = -1
        else:
            assoc[i] = j

    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''
    return ekf_state
    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###
    sigmaBar = ekf_state['P']
    # Establish Q
    # Q = np.zeros_like(Sigma)
    # Q[0, 0], Q[1, 1], Q[2, 2] = sigmas['xy'], sigmas['xy'], sigmas['phi']
    # for i in range(3, 3 + ekf_state['num_landmarks']):
    #     if i % 2 == 1:
    #         Q[i] = sigmas['range']
    #     else:
    #         Q[i] = sigmas['bearing']
    Q = np.eye(2)
    Q[0,0] = sigmas['range']
    Q[1,1] = sigmas['bearing']

    for i, tree in enumerate(trees):
        if assoc[i] == -1:
            initialize_landmark(ekf_state, tree)
            #new landmark made covariance grow
            sigmaBar = ekf_state['P']
        elif assoc[i] != -2:
            zhat, H = laser_measurement_model(ekf_state, i)
            zr, zb, _ = tree
            innov = np.array([zr-zhat[0], zb-zhat[1]])
            Kt = sigmaBar @H.transpose()@npl.inv(H @ sigmaBar @ H.transpose() + Q.transpose())
            ekf_state['x'] = ekf_state['x'] + Kt@innov
            ekf_state['P'] = slam_utils.make_symmetric( (np.eye(Kt.shape[0]) - Kt@H) @sigmaBar )



    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P']),
        'e': ['init']
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)
        state_history['e'].append(event[0])

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.4,
        "bearing": 3*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    state_history = run_ekf_slam(events[:1000], ekf_state, vehicle_params, filter_params, sigmas)
    t = PrettyTable(['Time', 'X', 'Y', 'Event'])
    with open('stateHistory.txt','w') as f:
        len_hist = len(state_history['t'])
        for i in range(len_hist):
            #line = '%.3f' % state_history['t'][i] + ',' + '%.2f' % state_history['x'][i][0] +',' + '%.2f' % state_history['x'][i][1]
            #f.write(line + "\n")
            t.add_row(['%.3f' % state_history['t'][i], '%.2f' % state_history['x'][i][0], '%.2f' % state_history['x'][i][1], state_history['e'][i]])
        f.write(str(t))

if __name__ == '__main__':
    main()
