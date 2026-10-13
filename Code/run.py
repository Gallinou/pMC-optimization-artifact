import subprocess
import time
import re

circles = {"red":"🔴","orange":"🟠","yellow":"🟡","green":"🟢","blue":"🔵","purple":"🟣","brown":"🟤","black":"⚫","white":"⚪"}

def print_time(time : float):
    if time == 0:
        return "no time"
    elif time < 1:
        return str(int(1000 * time)) + "ms"
    elif time < 60:
        return str(round(time, 2)) + "s"
    elif time < 3600:
        minutes = int(time // 60)
        seconds = round(time % 60, 2)
        return str(minutes) + "m" + str(seconds) + "s"
    else:
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = round(time % 60, 2)
        return str(hours) + "h" + str(minutes) + "m" + str(seconds) + "s"

def pMC_to_prism_prog(I, MF, ni, n, no, symbols, rewards, optimizer = None, nloop = None, nsample = None, dom = None, lr = None, lr_decay = None, weight_decay = None, alpha = None, beta = None, eps = None, gradient = None, timeout = None):
    assert ni == 1
    assert len(rewards) == no

    if optimizer is None:
        prism_prog = f"dtmc\n"
    else:
        assert optimizer in ["SGD", "SmallSGD", "AdaGrad", "Adam", "AdamW", "CAdamW", "CAdaGrad"]
        assert dom is not None

        if nloop is None:
            nloop = 0
        if timeout is None:
            timeout = 0

        assert nloop > 0 or timeout > 0
        
        if nsample is None:
            nsample = 0

        if optimizer == "SGD":
            if eps is None:
                eps = 0.001

            prism_prog = f"// SGD {nloop} {timeout} {nsample} {eps}"

        elif optimizer == "SmallSGD":
            if eps is None:
                eps = 0.001

            prism_prog = f"// SmallSGD {nloop} {timeout} {nsample} {eps}"

        elif optimizer == "AdaGrad":
            if lr is None:
                lr = 0.01

            if lr_decay is None:
                lr_decay = 0.0

            if weight_decay is None:
                weight_decay = 0.0

            if eps is None:
                eps = 1e-10

            prism_prog = f"// AdaGrad {nloop} {timeout} {nsample} {lr} {lr_decay} {weight_decay} {eps}"

        elif optimizer == "Adam":
            if lr is None:
                lr = 0.001

            if alpha is None:
                alpha = 0.9

            if beta is None:
                beta = 0.999

            if weight_decay is None:
                weight_decay = 0.0

            if eps is None:
                eps = 1e-8

            prism_prog = f"// Adam {nloop} {timeout} {nsample} {lr} {alpha} {beta} {weight_decay} {eps}"
            
        elif optimizer == "AdamW":
            if lr is None:
                lr = 0.001

            if alpha is None:
                alpha = 0.9

            if beta is None:
                beta = 0.999

            if weight_decay is None:
                weight_decay = 0.01

            if eps is None:
                eps = 1e-8

            prism_prog = f"// AdamW {nloop} {timeout} {nsample} {lr} {alpha} {beta} {weight_decay} {eps}"

        elif optimizer == "CAdamW":
            if lr is None:
                lr = 0.001

            if alpha is None:
                alpha = 0.9

            if beta is None:
                beta = 0.999

            if weight_decay is None:
                weight_decay = 0.01

            if eps is None:
                eps = 1e-8

            prism_prog = f"// CAdamW {nloop} {timeout} {nsample} {lr} {alpha} {beta} {weight_decay} {eps}"

        elif optimizer == "CAdaGrad":
            if lr is None:
                lr = 0.01

            if lr_decay is None:
                lr_decay = 0.0

            if weight_decay is None:
                weight_decay = 0.0

            if eps is None:
                eps = 1e-10

            prism_prog = f"// CAdaGrad {nloop} {timeout} {nsample} {lr} {lr_decay} {weight_decay} {eps}"

        if gradient is not None:
            assert gradient in ["LogDerivative", "SparseLU", "UniformPath", "BiCGSTAB", "JacobiIteration"]

            prism_prog += " "
            prism_prog += gradient

        prism_prog += f"\ndtmc// {n} {no} {len(symbols)}\n"
    
    for i, symbol in enumerate(symbols):
        prism_prog += f"const double {symbol}"
        prism_prog += " ;"
        if dom is not None:
            a, b = dom[i]
            if a is None or b is None:
                prism_prog += "// 0 1"
            else:
                prism_prog += f"// {a} {b}"
        prism_prog += "\n"
    prism_prog += f"\nmodule bernoulli\n    s : [0..{n + no + 1}] init 0;\n"

    prism_prog += "    [] s=0 -> "
    bool_first = True

    for (j, vj) in I[0]:
        if bool_first == True:
            bool_first = False
        else:
            prism_prog += " + "
        if type(vj) == str:
            prism_prog += vj
        else:
            prism_prog += str(vj.as_expr()).replace("**", "^").replace(" ", "")
        prism_prog += f" : (s'={j+1})"
    
    for i in range(n):
        prism_prog += f";\n    [] s={i+1} -> "
        bool_first = True
        for (j, vj) in MF[i]:
            if bool_first == True:
                bool_first = False
            else:
                prism_prog += " + "
            if type(vj) == str:
                prism_prog += vj
            else:
                prism_prog += str(vj.as_expr()).replace("**", "^").replace(" ", "")
            prism_prog += f" : (s'={j+1})"

    for i in range(no):
        prism_prog += f";\n    [] s={n+i+1} -> 1.0 : (s'={n+no+1})"

    prism_prog += ';\nendmodule\n\nrewards "score"\n'
    for i in range(no):
        prism_prog += f"    s={n+i+1} : {rewards[i]};\n"
    prism_prog += "endrewards"

    return prism_prog

def test_cpp(prism_prog: str, time_limit: float = None):
    st = time.time()
    try:
        # Run C++ program with an integrated timeout
        result = subprocess.run(
            ['./Code/main'],
            input=prism_prog,
            capture_output=True,
            text=True,
            timeout=time_limit
        )
        dt = time.time() - st
        return result.stdout, dt

    except subprocess.TimeoutExpired as e:
        dt = time.time() - st
        # Decode output if it was captured as bytes during timeout
        output = e.stdout
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return output, dt

    except Exception:
        dt = time.time() - st
        return None, dt

def parse_output(output : str) -> dict:
    result = {}

    result["error"] = "ERROR" in output

    init_match = re.search(r"Time for model initialization:\s*([\d.eE+-]+)", output)
    result["init_time_sec"] = float(init_match.group(1))

    grad_match = re.search(r"First gradient:\s*([^\n\r]+)", output)
    result["first_gradient"] = [float(x) for x in grad_match.group(1).split()]

    result["params"] = {}
    for p_match in re.finditer(
        r"(\w+)\s*=\s*(-?[\d.]+(?:[eE][+-]?\d+)?)", output
    ):
        name, val = p_match.groups()
        result["params"].setdefault(name, []).append(float(val))              

    if "score" in result["params"]:
        result["score"] = result["params"].pop("score")

    if "time" in result["params"]:
        result["opt_time_list"] = result["params"].pop("time")

    return result