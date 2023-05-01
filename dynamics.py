import numpy as np

class Dynamics:
    def __init__(self, m1, m2, l1, l2, g=9.8, static_torque=0.0):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.static_torque = static_torque

    def get_torques_alt(self, state):
        theta1 = state[0]
        theta2 = state[1] + state[0]

        tension = self.m2*self.g*np.cos(theta2)
        torque2 = -self.m2*self.g*np.sin(theta2)

        # ax, ay, bx, by = get_mass_positions(state)
        # a = ax + 1j*ay
        # b = bx + 1j*by

        return torque2

    def get_nominal_dynamics(self, state, action):
        # no friction
        cos = np.cos
        sin = np.sin
        Matrix = np.array
        
        theta1, theta2b, theta1_dot, theta2b_dot = state.flatten()
        tau1, tau2 = action.flatten()
        
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g

        # theta2b = theta2 - theta1
        # theta2b_dot = theta2_dot - theta1_dot
        
        B = Matrix([
            [l1**2*m1 + l1**2*m2 + l1*l2*m2*cos(theta2b), l1*l2*m2*cos(theta2b)],
            [l1*l2*m2*cos(theta2b) + l2**2*m2, l2**2*m2]
        ])

        g = Matrix([[-tau1 + g*l1*(m1+m2)*sin(theta1) - l1*l2*m2*(theta1_dot + theta2b_dot)**2 * sin(theta2b), 
                     theta1_dot**2*l1*l2*m2*sin(theta2b) - tau2 + g*l2*m2*sin(theta1 + theta2b)]]).T

        Binv = np.linalg.inv(B)

        thetasb_ddot = Binv @ (-g)

        # r is the angular accelerations that would occur
        # without static friction
        r = thetasb_ddot

        return r, Binv

    def get_dstate_dt(self, state, action, stictions=(0,0)):
        stictions = list(stictions)

        theta1, theta2b, theta1_dot, theta2b_dot = state.flatten()
        r, Binv = self.get_nominal_dynamics(state, action)

        if not any(stictions):
            theta1_ddot, theta2b_ddot = r.flatten()
            np.array((theta1_dot, theta2b_dot, theta1_ddot, theta2b_ddot)).reshape((4,1))

        s1, s2 = stictions

        # stiction model
        S = np.zeros((2,4))
        S[0,0] = 1-s1
        S[1,1] = 1-s2
        S[:,2:] = -Binv
        S[:,2] *= s1
        S[:,3] *= s2

        # r = S [phi1, phi2, stau1, stau2].T
        # S.t @ (S @ S.T).inv() r
        soln = S.T @ np.linalg.inv(S @ S.T) @ r
        phi1, phi2, stau1, stau2 = soln.flatten()
        staus = soln.flatten()[2:] # stiction torques

        # if max holding torque from frictions is turned off
        model_transition = False
        for i in range(2):
            if stictions[i] == 1:
                if abs(staus[i]) >= self.static_torque:
                    stictions[i] = 0
                    model_transition = True

        # debug
        debug = stictions[0] and not model_transition
        if debug:
            print(f"stiction active on 0!, staus {staus}")

        if model_transition:
            return self.get_dstate_dt(state, action, stictions)
        else:
            r_with_stiction, _ = self.get_nominal_dynamics(state, action.flatten() + staus)
            theta1_ddot, theta2b_ddot = r_with_stiction.flatten()

            dstate_dt = np.array((theta1_dot, theta2b_dot, theta1_ddot, theta2b_ddot)).reshape((4,1))

            if debug:
                print(f"dstate_dt with stiction {dstate_dt.T}")
                print(f"dstate_dt without stiction {dstate_dt.T}")
                
            return dstate_dt

    def get_mass_positions(self, state):
        theta1, theta2b = state.flatten()[:2]
        m1_pos = np.exp((theta1 - np.pi/2) * 1j) * self.l1
        m2_pos = np.exp((theta2b + theta1 - np.pi/2) * 1j) * self.l2 + m1_pos
        return np.array([m1_pos.real, m1_pos.imag, m2_pos.real, m2_pos.imag])

    def get_mass_velocities(self, state):
        theta1, theta2b, theta1_dot, theta2b_dot = state.flatten()
        m1_vel = 1j * np.exp((theta1 - np.pi/2) * 1j) * self.l1 * theta1_dot
        m2_vel = 1j * np.exp((theta2b + theta1 - np.pi/2) * 1j) * self.l2 * (theta1_dot + theta2b_dot) + m1_vel
        return np.array([m1_vel.real, m1_vel.imag, m2_vel.real, m2_vel.imag])

    def get_pe(self, state):
        mass_positions = self.get_mass_positions(state)
        min_h = -(self.l1 + self.l2)
        pe_alt = self.m1*self.g*(mass_positions[1] - min_h) + self.m2*self.g*(mass_positions[3] - min_h)
        return pe_alt

    def get_builtin_torques(self, state):
        dstate_dt = self.get_dstate_dt(state, np.zeros((2,1)))
        acc1, acc2 = dstate_dt.flatten()[2:]
        tau1 = self.m1*self.l1**2*acc1
        tau2 = self.m2*self.l2**2*acc2
        return tau1, tau2

    def get_ke(self, state):
        vels = self.get_mass_velocities(state)
        v1 = vels[:2]
        v2 = vels[2:]
        ke = 0.5*(self.m1*v1.T@v1 + self.m2*v2.T@v2).item()
        return ke

if __name__ == "__main__":
    print("Dynamics test")
    dynamics = Dynamics(m1=0.5, m2=0.1, l1=0.5, l2=0.4)
    state = np.array([0.5, 0.0, 0.0, 0.0]).reshape((4,1))

    action = np.array([0.2, 0.3])
    stau = np.array([1.209511083496357, -0.11206518886715236])
    print("resulting dynamics =\n", dynamics.get_dstate_dt(state, action, (1, 1)))
    # print("resulting dynamics w/ stau =\n", dynamics.get_dstate_dt(state, action + stau, (1, 1)))
