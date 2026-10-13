import numpy as np
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng()

states = ["i", "n", "x", "b"]

def M(p, q):
    state = "i"

    while True:
        if state == "i":
            n = rng.geometric(p)
            if n == 1:
                state = "b"
            else:
                state = "n"

        elif state == "n":
            x = rng.uniform(0, 1 / (n - 1))
            state = "x"

        elif state == "x":
            state = rng.choice(["+1", "b"], p=[x*q, 1-x*q])

        elif state == "b":
            state = rng.choice(["-1", "i"], p=[q, 1-q])

        if state == "+1":
            return 1
        elif state == "-1":
            return -1

def good_exit(p, q):
    return -np.log(p) * p * q / 2

def bad_exit(p, q):
    return (1 - good_exit(p, q)) * q

def avg_reward(p, q):
    g = good_exit(p, q)
    b = bad_exit(p, q)

    return (g - b) / (g + b)

def dM(p, q):
    state = "i"
    dlogp = 0
    dlogq = 0

    while True:
        if state == "i":
            n = rng.geometric(p)
            dlogp += (1 - n * p) / (p * (1 - p))
            if n == 1:
                state = "b"
            else:
                state = "n"

        elif state == "n":
            x = rng.uniform(0, 1 / (n - 1))
            state = "x"

        elif state == "x":
            state = rng.choice(["+1", "b"], p=[x*q, 1-x*q])
            if state == "+1":
                dlogq += 1 / q
            elif state == "b":
                dlogq += -x / (1 - x*q)

        elif state == "b":
            state = rng.choice(["0", "i"], p=[q, 1-q])
            if state == "0":
                dlogq += 1 / q
            elif state == "i":
                dlogq += -1 / (1 - q)

        if state == "+1":
            return dlogp, dlogq
        elif state == "0":
            return -dlogp, -dlogq

p, q = .5, .5

n_loop = 1000
n_sample = 10000
eps = 1e-2

lr = .001
alpha = 0.9
beta = 0.999
wd = .01
eps = 1e-8

mp, mq = 0, 0
vp, vq = 0, 0

print(f"Initial normalized reward: {avg_reward(p, q)} at p={p} and q={q}.")

YY = [avg_reward(p, q)]
XX = [(p, q)]
TT = [0]
t0 = time.time()
for i in range(n_loop):
    gp, gq = 0, 0
    for _ in range(n_sample):
        dlogp, dlogq = dM(p, q)
        gp += dlogp
        gq += dlogq

    gp /= n_sample
    gq /= n_sample

    p = (1 - lr * wd) * p
    q = (1 - lr * wd) * q

    mp = alpha * mp + (1 - alpha) * gp
    mq = alpha * mq + (1 - alpha) * gq

    vp = beta * vp + (1 - beta) * gp*gp
    vq = beta * vq + (1 - beta) * gq*gq

    mp_hat = mp / (1 - alpha**(i + 1))
    mq_hat = mq / (1 - alpha**(i + 1))

    vp_hat = vp / (1 - beta **(i + 1))
    vq_hat = vq / (1 - beta **(i + 1))

    p += lr * mp_hat / (np.sqrt(vp_hat) + eps)
    q += lr * mq_hat / (np.sqrt(vq_hat) + eps)

    if p >= 1:
        p = 1 - eps
    elif p <= 0:
        p = eps

    if q >= 1:
        q = 1 - eps
    elif q <= 0:
        q = eps

    YY.append(avg_reward(p, q))
    XX.append((p, q))
    TT.append(time.time() - t0)

print(f"Final normalized reward: {avg_reward(p, q)} at p={XX[-1][0]} and q={XX[-1][1]}.")
plt.plot(TT, YY)
plt.show()