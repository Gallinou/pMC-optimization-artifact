import numpy as np
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng()

states = ["i", "d", "c", "v"]

def M(p):
    state = "i"

    while True:
        if state == "i":
            state = rng.choice(["d", "c"])
        elif state == "d":
            x = np.sqrt(p)
            state = rng.choice(["-1", "i"], p=[x, 1-x])
        elif state == "c":
            x = 1/p
            state = rng.exponential(x)
        elif state == "+1":
            return 1
        elif state == "-1":
            return -1
        else:
            x = np.exp(-state/2)
            state = rng.choice(["+1", "i"], p=[x, 1-x])

def avg_reward(p):
    sp = np.sqrt(p)
    n = 4*sp
    d = 2*p + 2*sp + 1
    return n/d - 1

def dM(p):
    state = "i"
    result = 0

    while True:
        if state == "i":
            state = rng.choice(["d", "c"])
        elif state == "d":
            sp = np.sqrt(p)
            state = rng.choice(["-1", "i"], p=[sp, 1-sp])
            if state == "-1":
                d = 2*p
                result += 1/d
            elif state == "i":
                d = 2*sp*(1 - sp)
                result += -1/d
        elif state == "c":
            beta = 1/p
            state = rng.exponential(beta)
            v = beta - state
            result += v
        elif state == "+1":
            return result
        elif state == "-1":
            return -result
        else:
            x = np.exp(-state/2)
            state = rng.choice(["+1", "i"], p=[x, 1-x])

def avg_reward_derivative(p):
    sp = np.sqrt(p)
    n = 2 - 4*p
    a = 2*p + 2*sp + 1
    d = a*a*sp
    return n/d

n_loop = 1000
n_sample = 10000
eps = 1e-2
p = 1

lr = .001
alpha = 0.9
beta = 0.999
wd = .01
eps = 1e-8

m, v = 0, 0

print(f"Initial normalized reward: {avg_reward(p)} at p={p}.")

YY = [avg_reward(p)]
XX = [p]
TT = [0]
t0 = time.time()
for i in range(n_loop):
    g = 0
    for _ in range(n_sample):
        g += dM(p)
    g /= n_sample

    p = (1 - lr * wd) * p

    m = alpha * m + (1 - alpha) * g
    v = beta * v  + (1 - beta)  * g*g

    m_hat = m / (1 - alpha**(i + 1))
    v_hat = v / (1 - beta **(i + 1))

    p += lr * m_hat / (np.sqrt(v_hat) + eps)

    if p >= 1 or p <= 0:
        break
    else:
        YY.append(avg_reward(p))
        XX.append(p)
        TT.append(time.time() - t0)

print(f"Final normalized reward: {avg_reward(p)} at p={XX[-1]}.")
plt.plot(TT, YY)
plt.show()