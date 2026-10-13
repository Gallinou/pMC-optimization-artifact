#include "basic.hpp"

SparseMarkovModel GenerateSparceMarkovModel() {
    Variables vars;
    vector<Interval> intervals;
    SparseMatrix transitions, transitionsSUM;
    SparseTensor dtransitions;
    Rewards rewards;
    Point p;
    int n, no, nv, nt, j, nloop, nsample, ntrans = 0;
    double val, a, b, timeout;
    double lr = 0.0, lr_decay = 0.0, weight_decay = 0.0, alpha = 0.0, beta = 0.0, eps = 0.0;
    string s;
    OptimizerType optimizer = OptimizerType::AdamW;
    GradientMethod gradientmethod = GradientMethod::SparseLU;

    // // {optimizer} {nloop} {nsample} {gradientmethod}
    cin >> s >> s >> nloop >> timeout >> nsample;
    if (s == "SGD") { // eps
        optimizer = OptimizerType::SGD;
        cin >> eps;
        cout << "SGD(eps:" << eps << ")" << endl;
    }
    else if (s == "SmallSGD") { // eps
        optimizer = OptimizerType::SmallSGD;
        cin >> eps;
        cout << "SmallSGD(eps:" << eps << ")" << endl;
    }
    else if (s == "AdaGrad") { // lr lr_decay weight_decay eps
        optimizer = OptimizerType::AdaGrad;
        cin >> lr >> lr_decay >> weight_decay >> eps;
        cout << "AdaGrad(lr:" << lr << ", lr_decay:" << lr_decay << ", weight_decay:" << weight_decay << ", eps:" << eps << ")" << endl; 
    }
    else if (s == "Adam") { // lr alpha beta weight_decay eps
        optimizer = OptimizerType::Adam;
        cin >> lr >> alpha >> beta >> weight_decay >> eps;
        cout << "Adam(lr:" << lr << ", alpha:" << alpha << ", beta:" << beta << ", weight_decay:" << weight_decay << ", eps:" << eps << ")" << endl;
    }
    else if (s == "AdamW") { // lr alpha beta weight_decay eps
        optimizer = OptimizerType::AdamW;
        cin >> lr >> alpha >> beta >> weight_decay >> eps;
        cout << "AdamW(lr:" << lr << ", alpha:" << alpha << ", beta:" << beta << ", weight_decay:" << weight_decay << ", eps:" << eps << ")" << endl;
    }
    else if (s == "CAdamW") { // lr alpha beta weight_decay eps
        optimizer = OptimizerType::CAdamW;
        cin >> lr >> alpha >> beta >> weight_decay >> eps;
        cout << "CAdamW(lr:" << lr << ", alpha:" << alpha << ", beta:" << beta << ", weight_decay:" << weight_decay << ", eps:" << eps << ")" << endl;
    }
    else if (s == "CAdaGrad") { // lr lr_decay weight_decay eps
        optimizer = OptimizerType::CAdaGrad;
        cin >> lr >> lr_decay >> weight_decay >> eps;
        cout << "CAdaGrad(lr:" << lr << ", lr_decay:" << lr_decay << ", weight_decay:" << weight_decay << ", eps:" << eps << ")" << endl; 
    }

    cin >> s;
    if (s == "SparseLU")
        gradientmethod = GradientMethod::SparseLU;
    else if (s == "LogDerivative")
        gradientmethod = GradientMethod::LogDerivative;
    else if (s == "UniformPath")
        gradientmethod = GradientMethod::UniformPath;
    else if (s == "BiCGSTAB")
        gradientmethod = GradientMethod::BiCGSTAB;
    else if (s == "JacobiIteration")
        gradientmethod = GradientMethod::JacobiIteration;

    // dtmc// {n} {no} {nv}
    cin >> s >> n >> no >> nv;

    // const double {var} ;// a b
    for (int i = 0; i < nv; i++) {
        cin >> s >> s >> s;
        vars.emplace_back(s);
        cin >> s;
        p.emplace_back(.5);
        cin >> a >> b;
        intervals.emplace_back(a, b);
    }

    // module bernoulli
    getline(cin, s);
    getline(cin, s);
    getline(cin, s);

    // s : [0..{nt}] init 0;
    nt = n + no + 1;
    getline(cin, s);

    // [] s={i} -> {p1} : (s'={j1}) + ... + {pm} : (s'={jm});
    transitions.resize(n + 1);
    transitionsSUM.resize(n + 1);
    dtransitions.resize(n + 1);
    for (int i = 0; i < n + 1; i++) {
        dtransitions[i].resize(nv);
    }
    for (int i = 0; i < n + 1; i++) {
        cin >> s >> s;
        assert((cin >> s && s == "->"));
        Expression polySUM(0);
        while (true) {
            cin >> s;
            Expression poly(s);
            polySUM = polySUM + poly;
            cin >> s >> s;
            assert(sscanf(s.c_str(), "(s'=%d)", &j));
            transitions[i].emplace_back(Entry(j, poly));
            transitionsSUM[i].emplace_back(Entry(j, polySUM));
            for (int k = 0; k < nv; k++)
                dtransitions[i][k].emplace_back(Entry(j, poly.diff(vars[k])));
            ntrans++;
            if (s.back() == ';') break;
            cin >> s;
        }
    }
    for (int i = 0; i < no + 2; i++) {
        getline(cin, s);
    }

    // endmodule
    
    // rewards "score"
    cin >> s >> s;

    // s={i} : {val}
    for (int i = n + 1; i < nt; i++) { 
        cin >> s >> s >> s;
        sscanf(s.c_str(), "%lf;", &val);
        rewards.emplace_back(val);
    }

    //endrewards

    return SparseMarkovModel(n + 1, nv, nloop, timeout, nsample, ntrans, optimizer, gradientmethod, lr, lr_decay, weight_decay, alpha, beta, eps, vars, intervals, transitions, transitionsSUM, dtransitions, rewards, p);
}