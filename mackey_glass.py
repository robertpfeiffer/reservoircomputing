import numpy
import random

def mackey_glass(beta, gamma, n, tau, x, dt):
    # ring buffer
    buf_size = int(tau/dt)
    buf      = [abs(random.gauss(0.5,0.1)) for a in range(buf_size)]
    buf_idx  = 0
    while True:
        x_tau = buf[buf_idx]
        dx_dt = beta *x_tau/(1+x_tau**n) - gamma*x
        x = x + dx_dt*dt
        buf[buf_idx] = x
        buf_idx = (1 + buf_idx) % buf_size
        yield x
