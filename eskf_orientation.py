
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class OrientationFilter:
    def __init__(self):
        self.q = np.array([1, 0, 0, 0])  # Identity quaternion
        self.omega = np.zeros(3)
        self.d_theta = np.zeros(3) # error state
        self.d_omega = np.zeros(3)
        self.R = 1e-3*np.eye(4)
        self.Q = 1e-3*np.eye(6)
        self.P = np.eye(6)
 
    
    def _get_scalar(self, q):
        return q[0]
    
    def _get_vector(self, q):
        return q[1:]
    
    def left_quat_matrix(self, q):
        """
        Return the 4x4 matrix Q such that Q @ q2 == quat_left_product(q, q2)
        q must be in [w, x, y, z] format.
        """
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    def right_quat_matrix(self, q):
        """
        Return the 4x4 matrix R such that R @ q1 == quat_right_product(q1, q)
        q must be in [w, x, y, z] format.
        """
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w,  z, -y],
            [y, -z,  w,  x],
            [z,  y, -x,  w]
        ])
    
    def quat_inverse(self, q):
        """
        Gibt das Inverse eines Quaternions zurück.
        q: Quaternion im Format [w, x, y, z]
        """
        q = np.asarray(q)
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    def quat_exp_pure(self, v):
        """
        Quaternion exponential of a pure quaternion [0, x, y, z].
        Input: v -- 3D numpy array (vector part)
        Output: 4D numpy array representing quaternion [w, x, y, z]
        """
        theta = np.linalg.norm(v)
        if theta < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            w = np.cos(theta)
            xyz = np.sin(theta) * (v / theta)
            return np.concatenate(([w], xyz))
        
    def skew(self, v):
        """
        Return the 3x3 skew-symmetric matrix of a 3D vector v.
        v must be a numpy array of shape (3,)
        """
        x, y, z = v
        return np.array([
            [ 0, -z,  y],
            [ z,  0, -x],
            [-y,  x,  0]
        ])
        

    def _compute_H(self,q):
        dq = self.quat_exp_pure(self.d_theta)
        dH_dq = self.right_quat_matrix(dq)
        dH_dw = np.zeros((4,3))
        Hx = np.block([dH_dq,dH_dw])
        Ql = self.left_quat_matrix(q)
        Xq_dtheta = 1/2*Ql@np.block([
                    [np.zeros((1, 3))],  # 1x3 row of zeros
                    [np.eye(3)]          # 3x3 identity
                ])
        Xq_dw = np.zeros((4,3))
        Xw_dtheta = np.zeros((3,3))
        Xw_dw = np.eye(3)
        X_dx = np.block([[Xq_dtheta,Xq_dw],[Xw_dtheta,Xw_dw]])
        return Hx@X_dx


    
    def predict(self, dt):
        F_d_theta = -self.skew(self.omega)*dt
        Fi_d_theta = np.eye(3)
        F_omega = np.eye(3)
        Fi_omega = np.eye(3)
        Fx = np.block([
            [F_d_theta,      np.zeros((3, 3))],
            [np.zeros((3, 3)), F_omega      ]
        ])
        Fi = np.block([
            [Fi_d_theta,      np.zeros((3, 3))],
            [np.zeros((3, 3)), Fi_omega      ]
        ])

        xplus = Fx@np.concatenate([self.d_theta, self.d_omega])
        self.d_theta = xplus[:3]
        self.d_omega = xplus[3:]
        P = self.P
        self.P = Fx@P@Fx.T + Fi@self.Q@Fi.T

    def update(self, q_meas):
        P = self.P
        H = self._compute_H(self.q)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + self.R)
        q_e = self.left_quat_matrix(q_meas) @ self.quat_inverse(self.q)
        x_post = K@(q_e)
        self.d_theta = x_post[:3]
        self.d_omega = x_post[3:]

        self.P = (np.eye(6) - K@H)@P

    def error_injection(self):
        # error injection
        self.q = self.left_quat_matrix(self.q) @ self.quat_exp_pure(self.d_theta)
        self.omega = self.omega + self.d_omega
        
        # error reset
        self.d_theta = np.zeros(3) 
        self.d_omega = np.zeros(3)
        P = self.P
        G = np.eye(6) # np.eye(3) - self.skew(1/2*self.d_theta) # TODO: update with correct term
        self.P = G@P@G.T
    
    def step(self,dt,q_meas):
        self.predict(dt)
        self.update(q_meas)
        self.error_injection()
        print("q: {}, w: {}".format(self.q, self.omega))
