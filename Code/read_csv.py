import csv
import math
from ast import literal_eval

file_name = "example.csv"

def cosine(l1, l2):
    assert len(l1) == len(l2)

    n1 = math.sqrt(sum([x**2 for x in l1]))
    n2 = math.sqrt(sum([x**2 for x in l2]))
    n = sum([x * y for x, y in zip(l1, l2)])

    return n / (n1 * n2)

def norm_l2(l1, l2):
    assert len(l1) == len(l2)

    n1 = math.sqrt(sum([x**2 for x in l1]))
    n = math.sqrt(sum([(x - y)**2 for x, y in zip(l1, l2)]))

    return n / n1

def add_item_aux(dic, key, val):
    if key not in dic:
        dic[key] = (0, 0.0)
    i, current_val = dic[key]
    dic[key] = (i + 1, current_val + val)

key_list = ["time", "norm_time", "cosine", "norm_dist"]
nstate_dic = {key:{} for key in key_list}
nvar_dic = {key:{} for key in key_list}
id_dic = {key:{} for key in key_list}
avg_rewards = {}

def add_item(key, nstate, nvar, id, val):
    add_item_aux(nstate_dic[key], (id, nstate), val)
    add_item_aux(nvar_dic[key], (id, nvar), val)
    add_item_aux(id_dic[key], id, val)

len_rewards = 0
def add_rewards(rewards, times, rmax, id):
    assert len(rewards) <= len(times)

    if id not in avg_rewards:
        avg_rewards[id] = [(0, 0.0, 0.0)] * len_rewards

    for i, (r, t) in enumerate(zip(rewards, times)):
        j, avg_r, avg_t = avg_rewards[id][i]
        if rmax == 0.0:
            avg_rewards[id][i] = (j + 1, avg_r + 0.0, avg_t + t) 
        else:
            avg_rewards[id][i] = (j + 1, avg_r + max(r, 0.0) / rmax, avg_t + t)  

    r, t = rewards[len(rewards) - 1], times[len(rewards) - 1]
    for i in range(len(rewards), len(avg_rewards[id])):
        j, avg_r, avg_t = avg_rewards[id][i]
        if rmax == 0.0:
            avg_rewards[id][i] = (j + 1, avg_r + 0.0, avg_t + t) 
        else:
            avg_rewards[id][i] = (j + 1, avg_r + max(r, 0.0) / rmax, avg_t + t)  

with open(file_name, mode="r", encoding="utf-8") as file:
    experiments = list(csv.DictReader(file))

for i in range(len(experiments)):
    for data_key in experiments[i]:
        if data_key not in ["method:optimizer", "method:gradient", "opt:status"]:
            try:
                experiments[i][data_key] = literal_eval(experiments[i][data_key])
                if data_key == "opt:times" and type(experiments[i][data_key]) == float:
                    experiments[i][data_key] = [experiments[i][data_key]]
                if data_key == "opt:rewards":
                    len_rewards = max(len_rewards, len(experiments[i][data_key]))
            except:
                experiments[i][data_key] = None

imax = 0
all_nstate, all_nvar = set(), set()
while imax < len(experiments):
    id = experiments[imax]["graph:id"]
    imin = imax
    sparseLU_grad, sparseLU_time, max_reward = None, None, 0.0
    while imax < len(experiments) and experiments[imax]["graph:id"] == id:
        if experiments[imax]["method:gradient"] == "SparseLU":
            sparseLU_grad = experiments[imax]["opt:first_gradient"]
            if experiments[imax]["opt:times"] is not None and len(experiments[imax]["opt:times"]) > 0:
                sparseLU_time = experiments[imax]["opt:times"][-1]
        if experiments[imax]["opt:rewards"] is not None and len(experiments[imax]["opt:rewards"]) > 0:
            reward = experiments[imax]["opt:rewards"][-1] - experiments[imax]["opt:rewards"][0]
            max_reward = max(reward, max_reward)
        imax += 1

    for i in range(imin, imax):
        nstate = experiments[i]["graph:nstate"]
        ntrans = experiments[i]["graph:ntrans"]
        nvar   = experiments[i]["graph:nvar"]
        id     = experiments[i]["method:id"]

        times    = experiments[i]["opt:times"]
        rewards  = experiments[i]["opt:rewards"]
        gradient = experiments[i]["opt:first_gradient"]

        all_nstate.add(nstate)
        all_nvar.add(nvar)

        if times is not None and len(times) > 0:
            add_item("time", nstate, nvar, id, times[-1])

            if sparseLU_time is not None and sparseLU_time != 0.0:
                add_item("norm_time", nstate, nvar, id, times[-1] / sparseLU_time)

        if sparseLU_grad is not None and len(sparseLU_grad) > 0 and gradient is not None and len(gradient) > 0:
            add_item("cosine", nstate, nvar, id, cosine(sparseLU_grad, gradient))
            add_item("norm_dist", nstate, nvar, id, norm_l2(sparseLU_grad, gradient))

        if rewards is not None and len(rewards) > 0:
            add_rewards(rewards, times, max_reward, id)

for key in key_list:
    if len(all_nstate) > 1:
        data_dic = {}
        print(f"--- {key} per nstate ---")
        for (id, nstate) in nstate_dic[key]:
            i, v = nstate_dic[key][(id, nstate)]
            if id not in data_dic:
                data_dic[id] = [(nstate, v / i)]
            else:
                data_dic[id].append((nstate, v / i))
        for id in sorted(data_dic.keys()):
            print(id, end=": ")
            for nstate, v in data_dic[id]:
                print(f"({nstate},{v})", end="")
            print()

    if len(all_nvar) > 1:
        data_dic = {}
        print(f"--- {key} per nvar ---")
        for (id, nvar) in nvar_dic[key]:
            i, v = nvar_dic[key][(id, nvar)]
            if id not in data_dic:
                data_dic[id] = [(nvar, v / i)]
            else:
                data_dic[id].append((nvar, v / i))
        for id in sorted(data_dic.keys()):
            print(id, end=": ")
            for nvar, v in data_dic[id]:
                print(f"({nvar},{v})", end="")
            print()

    print(f"--- {key} ---")
    for id in sorted(id_dic[key].keys()):
        i, v = id_dic[key][id]
        print(f"{id}: {v / i}")
    print()

if len_rewards > 0:
    print("--- norm_reward per time ---")
    for id in sorted(avg_rewards.keys()):
        print(id, end=": ")
        for (i, r, t) in avg_rewards[id]:
            print(f"({t / i},{r / i})", end="")
        print()