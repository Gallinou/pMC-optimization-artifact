import Code.test as test
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run experiments with custom N, K lists, and step.")
    
    # Settings
    parser.add_argument('-N', '--n_list', nargs='+', type=int, default=[10], 
                        help='List of integers for the number of states (e.g. -N 10 20 30)')
    parser.add_argument('-K', '--k_list', nargs='+', type=int, default=[1], 
                        help='List of integers for the number of variables (e.g. -K 1 2 3)')
    parser.add_argument('-P', '--p_list', nargs='+', type=float, default=[None], 
                        help='List of float for the proportion of transitions (e.g. -P .2 .5)')
    parser.add_argument('-nstep', '--n_step', type=int, default=None, 
                        help='Integer for the number of steps (e.g. -nstep 10)')
    parser.add_argument('-ngraph', '--n_graph', type=int, default=20, 
                        help='Integer for the number of Markov chains (e.g. -ngraph 20)')
    parser.add_argument('-st', '--soft_timeout', type=float, default=None,
                        help='Maximum time before c++ optimizer timeout (e.g. -st 180)')
    parser.add_argument('-ht', '--hard_timeout', type=float, default=180.0,
                        help='Maximum time before python multiprocessing timeout (e.g. -ht 180)')
    parser.add_argument('-show', action='store_true', 
                        help='Enable circle visualization (default: False)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=None,
                        help='Learning rate (e.g. -lr .001)')
    parser.add_argument('-zip', action='store_true', 
                        help='Zip the number of states and the number of variables (default: False)')

    # Optimizers
    parser.add_argument('-SGD', action='store_true',
                        help='Add SGD to the list of optimizers')
    parser.add_argument('-SmallSGD', action='store_true',
                        help='Add SmallSGD to the list of optimizers')
    parser.add_argument('-AdaGrad', action='store_true',
                        help='Add AdaGrad to the list of optimizers')
    parser.add_argument('-Adam', action='store_true',
                        help='Add Adam to the list of optimizers')
    parser.add_argument('-AdamW', action='store_true',
                        help='Add AdamW to the list of optimizers')
    parser.add_argument('-CAdamW', action='store_true',
                        help='Add CAdamW to the list of optimizers')
    parser.add_argument('-CAdaGrad', action='store_true',
                        help='Add CAdaGrad to the list of optimizers')

    # Methods
    parser.add_argument('-SparseLU', action='store_true',
                        help='Run the SparseLU gradient method')
    parser.add_argument('-BiCGSTAB', action='store_true',
                        help='Run the BiCGSTAB gradient method')
    parser.add_argument('-LogDerivative', '--LD_list', nargs='+', type=int, default=[],
                        help='Run the LogDerivative gradient method (e.g. -LogDerivative 100 1000 10000)')
    parser.add_argument('-UniformPath', '--UP_list', nargs='+', type=int, default=[],
                        help='Run the UniformPath gradient method (e.g. -UniformPath 100 1000 10000)')
    parser.add_argument('-JacobiIteration', '--JI_list', nargs='+', type=int, default=[],
                        help='Run the JacobiIteration gradient method (e.g. -JacobiIteration 100 1000 10000)')

    args = parser.parse_args()

    # Extract values
    N = args.n_list
    K = args.k_list
    P = args.p_list
    nloop = args.n_step
    ngraph = args.n_graph
    show_circle = args.show
    soft_timeout = args.soft_timeout
    hard_timeout = args.hard_timeout
    lr = args.learning_rate
    all_optimizers = ["SGD", "SmallSGD", "AdaGrad", "Adam", "AdamW", "CAdamW", "CAdaGrad"]
    optimizers = [opt for opt in all_optimizers if getattr(args, opt)]
    
    if optimizers == []:
        optimizers = ["AdamW"]

    print(f"Loaded n_step: {nloop}")
    print(f"Loaded n_graph: {ngraph}")
    print(f"Loaded soft_timeout: {soft_timeout}")
    print(f"Loaded hard_timeout: {hard_timeout}")
    print(f"Loaded optimizer: {optimizers}")

    print("\nSettings:")
    if args.zip:
        assert len(N) == len(K), "N and K should be of the same length"
        settings = [(ngraph, n, k, p) for (n, k) in zip(N, K) for p in P]
    else:
        settings = [(ngraph, n, k, p) for n in N for k in K for p in P]
    for ngraph, n, k, p in settings:
        print(f"ngraph={ngraph} n={n} k={k} p={p}")

    assert len(settings) > 0, "no setting selected"

    #optimizer, nsample, gradient(LogDerivative, SparseLU, UniformPath, BiCGSTAB, JacobiIteration)
    methods = []
    for optimizer in optimizers:
        if args.SparseLU == True:methods.append([optimizer, lr, None, "SparseLU"])
        if args.BiCGSTAB == True: methods.append([optimizer, lr, None, "BiCGSTAB"])
        for nsample in args.LD_list: methods.append([optimizer, lr, nsample, "LogDerivative"])
        for nsample in args.UP_list: methods.append([optimizer, lr, nsample, "UniformPath"])
        for nsample in args.JI_list: methods.append([optimizer, lr, nsample, "JacobiIteration"])

    assert len(methods) > 0, "no method selected"
    assert nloop is not None or soft_timeout is not None, "unbounded optimization"

    print("\nMethods:")
    if len(methods) != 0:
        for optimizer, lr, nsample, gradient in methods:
            print(f"optimizer={optimizer} lr={lr} nsample={nsample} gradient={gradient}")
        print()

        test.multi_generate(settings, methods, rewards=[1, -1], nloop=nloop, soft_timeout=soft_timeout, hard_timeout=hard_timeout, show_circle=show_circle)

    else:
        print("None")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"> Error: {e}")