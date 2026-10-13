import Code.run as run
import Code.generator as gen
import time
import csv

def print_circle(color : str, show : bool = True):
    if show == True:
        print(run.circles[color], end="", flush=True)

def is_monotonic(params : dict):
    for l in list(params.values()):
        sign = 0
        for i in range(len(l) - 1):
            if l[i + 1] - l[i] > 0:
                if sign == -1:
                    return False
                sign = 1

            elif l[i + 1] - l[i] < 0:
                if sign == 1:
                    return False
                sign = -1
    return True

def test(prism_prog : str, soft_timeout : float, hard_timeout : float, nloop : int, show_circle : bool):
    out, _ = run.test_cpp(prism_prog, hard_timeout)

    try:
        result = run.parse_output(out)

        opt_time_list = result["opt_time_list"]
        optimization_time = opt_time_list[-1]
        first_grad = result["first_gradient"]

        if nloop == 1:
            print_circle("blue", show_circle)
            return True, None, None, first_grad, optimization_time
        else:
            params = result["params"]
            initial_reward = result["score"][0]
            final_reward = result["score"][-1]

            scores = [x - initial_reward for x in result["score"]]

            if soft_timeout is not None and optimization_time > soft_timeout:
                print_circle("black", show_circle)
                return None, scores, params, first_grad, opt_time_list
            
            elif final_reward > initial_reward:
                print_circle("green", show_circle)
                return True, scores, params, first_grad, opt_time_list
            
            elif final_reward == initial_reward:
                print_circle("yellow", show_circle)
                return False, scores, params, first_grad, opt_time_list
            
            else:
                print_circle("red", show_circle)
                return False, scores, params, first_grad, opt_time_list

    except:
        print_circle("black", show_circle)
        return None, [], {}, [], []

def generate(ngraph, nstate, nvar, p, rewards, methods, nloop, soft_timeout, hard_timeout, show_circle : bool = True, writer = None, id_min : int = 0):
    symbols = gen.list_variables(nvar)

    #Optimization
    t0 = time.time()
    current_id = id_min
    for i in range(ngraph):
        while True:
            try:
                if p is None:
                    I, MF, dom, ntrans = gen.random_simple_pMC(0, 1, nstate, 2, -.1, .1, symbols, fast=True)
                else:
                    I, MF, dom, ntrans = gen.random_simple_pMC(p, 1, nstate, 2, -.1, .1, symbols, fast=False)
                break
            except:
                pass

        if show_circle == True:
            print(f"Graph {i+1}: ", end="")

        for j, (optimizer, lr, nsample, gradient) in enumerate(methods):
            try:
                prism_prog = run.pMC_to_prism_prog(I, MF, 1, nstate, 2, symbols, rewards, optimizer=optimizer, nloop=nloop, nsample=nsample, dom=dom, gradient=gradient, lr=lr, timeout=soft_timeout)
                opt_status, opt_scores, opt_params, opt_first_gradient, opt_time_list = test(prism_prog, soft_timeout, hard_timeout, nloop, show_circle)

                if writer is not None:
                    str_opt_status = "Success" if opt_status == True else "Fail" if opt_status == False else "Timeout"
                    opt_monotonic = None if nloop == 1 else is_monotonic(opt_params)
                    writer.writerow([current_id, nstate, ntrans, nvar, j, optimizer, nsample, gradient, str_opt_status, opt_time_list, opt_monotonic, opt_scores, opt_first_gradient])

            except:
                print_circle("white", show_circle)

        if show_circle == True:
            print()

        current_id += 1

    dt = time.time() - t0
    print(f"Optimization in {run.print_time(dt)}\n")

    return current_id

def multi_generate(settings, methods, rewards, nloop, soft_timeout, hard_timeout, show_circle : bool = True):
    csv_file = open('output.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(["graph:id", "graph:nstate", "graph:ntrans", "graph:nvar", "method:id", "method:optimizer", "method:nsample", "method:gradient", "opt:status", "opt:times", "opt:monotonic", "opt:rewards", "opt:first_gradient"])

    current_id = 0
    for i, (ngraph, nstate, nvar, p) in enumerate(settings):
        print(f"> Setting {i + 1}: ngraph={ngraph} nstate={nstate} nvar={nvar} p={p}")

        current_id = generate(ngraph, nstate, nvar, p, rewards, methods, nloop, soft_timeout, hard_timeout, show_circle, writer, current_id)
        csv_file.flush()

    csv_file.close()