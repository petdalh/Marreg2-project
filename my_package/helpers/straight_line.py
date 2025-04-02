import numpy as np

# Temporary configurations
p0 = np.array([0.0, 0.0])
p1 = np.array([5.0, 0.0])
psi_d_const = 0.0
U_ref = 0.5
mu = 0.1
eps = 1e-6

# Constant heading if psi_ref not given:
psi_d_const = 0

def straight_line(s):

   # Position on line
   pd = (1 - s)*p0 + s*p1
   # Heading
   psi_d = psi_d_const

   
   eta_d = np.array([pd[0], pd[1], psi_d])

   # 2) First derivative wrt s
   pd_s = (p1 - p0)
   psi_d_s = 0.0
   eta_ds = np.array([pd_s[0], pd_s[1], psi_d_s], dtype=float)

   # 3) Second derivative is zero for straight line
   eta_ds2 = np.zeros_like(eta_ds)

   return eta_d, eta_ds, eta_ds2

def update_law(observation, s):
   # task 4.5 normalized gradient update law
   if observation.eta == None:
      eta_hat = np.zeros((3, 1))   
   else:
      eta_hat = np.array(observation.eta).reshape(3, 1)

   p_hat = eta_hat[:2, 0]  
   eta_d, eta_ds, eta_ds2 = straight_line(s)

   norm_eta_ds = np.linalg.norm(eta_ds[:2]) + eps


   p_d = eta_d[:2]
   V1_s = -(p1 - p0).T@(p_hat - p_d)

   w = - mu/(norm_eta_ds)* V1_s
   v_s = U_ref / np.linalg.norm(p1 - p0)
   v_ss = 0.0


   return w, v_s, v_ss
