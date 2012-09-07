import numpy
import random
import collections

def mackey_glass(beta, gamma, n, tau, x, dt):
    buf_size = int(tau/dt)
    buf      = collections.deque([abs(random.gauss(0.0,0.1)) for a in range(buf_size)],buf_size)
    while True:
        x_tau = buf[0]
        dx_dt = beta * x_tau/(1.0+x_tau**n) - gamma*x
        x = x + dx_dt * dt
        buf.append(x)
        yield x
