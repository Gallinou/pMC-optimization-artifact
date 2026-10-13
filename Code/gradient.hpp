#include "optimizer.hpp"

bool SparseLUUpdate(SparseMarkovModel& model) {
    // 1. Build (I - A)
    vector<Eigen::Triplet<double>> triplets;
    int N = model.n + model.nout;
    triplets.reserve(model.ntrans + N);

    for (int i = 0; i < N; i++) // I
        triplets.emplace_back(i, i, 1.0);

    for (int i = 0; i < model.n; i++) // -transitions
        for (const auto& e : model.transitions[i])
            triplets.emplace_back(i, e.id, -eval_expr(e.expr, model.subs));

    Eigen::SparseMatrix<double> IminusA(N, N);
    IminusA.setFromTriplets(triplets.begin(), triplets.end());

    // 2. Build b
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
    for (int i = 0; i < model.nout; i++)
        b[model.n + i] = model.rewards[i];

    // 3. Solve the linear equation globally
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    IminusA.makeCompressed();
    solver.analyzePattern(IminusA);
    solver.factorize(IminusA);
    if (solver.info() != Eigen::Success) return false;

    // 4. Build x := (I - A)^-1 * b
    Eigen::VectorXd x = solver.solve(b);

    // 5. Build dx := (I - A)^-1 * dA * x
    Eigen::VectorXd c(N), dx(N);
    Gradient g(model.k);
    for (int k = 0; k < model.k; k++) {
        c.setZero();

        for (int i = 0; i < model.n; i++) // Build c := dA * x
            for (const auto& e : model.dtransitions[i][k])
                c[i] += eval_expr(e.expr, model.subs) * x[e.id];

        dx = solver.solve(c);
        g[k] = dx[0]; // gradient of the expected reward from the initial state
    }

    return optimize(model, g);
}

bool BiCGSTABUpdate(SparseMarkovModel& model) {
    // 1. Build (I - A)
    vector<Eigen::Triplet<double>> triplets;
    int N = model.n + model.nout;
    triplets.reserve(model.ntrans + N);

    for (int i = 0; i < N; i++) // I
        triplets.emplace_back(i, i, 1.0);

    for (int i = 0; i < model.n; i++) // -transitions
        for (const auto& e : model.transitions[i])
            triplets.emplace_back(i, e.id, -eval_expr(e.expr, model.subs));

    Eigen::SparseMatrix<double> IminusA(N, N);
    IminusA.setFromTriplets(triplets.begin(), triplets.end());

    // 2. Build b
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
    for (int i = 0; i < model.nout; i++)
        b[model.n + i] = model.rewards[i];

    // 3. Solve the linear equation globally using BiCGSTAB
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    IminusA.makeCompressed();
    solver.analyzePattern(IminusA);
    solver.factorize(IminusA);
    if (solver.info() != Eigen::Success) return false;

    // 4. Build x := (I - A)^-1 * b
    Eigen::VectorXd x = solver.solve(b);

    // 5. Build dx := (I - A)^-1 * dA * x
    Eigen::VectorXd c(N), dx(N);
    Gradient g(model.k);
    for (int k = 0; k < model.k; k++) {
        c.setZero();

        for (int i = 0; i < model.n; i++) // Build c := dA * x
            for (const auto& e : model.dtransitions[i][k])
                c[i] += eval_expr(e.expr, model.subs) * x[e.id];

        dx = solver.solve(c);
        g[k] = dx[0]; // gradient of the expected reward from the initial state
    }

    return optimize(model, g);
}

int SampleIndex(int s, SparseMarkovModel& model) {
    double x = rng();
    int a = 0, b = model.transitionsSUM[s].size();

    while (a < b) {
        int m = a + (b - a) / 2;
        if (eval_entry(model.transitionsSUM[s][m], model.subs, model.version) < x) {
            a = m + 1;
        }
        else b = m;
    }

    return a;
}

bool LogDerivativeUpdate(SparseMarkovModel& model) {
    int s, i;
    double w, reward;
    Weights ws(model.k, 0.0);
    Gradient g;
    
    g.resize(model.k, 0.0);

    double avg_reward = 0.0;
    for (int n = 0; n < model.nsample; n++) {
        fill(ws.begin(), ws.end(), 0.0);
        s = 0;
        while (s < model.n) {
            i = SampleIndex(s, model);
            w = eval_entry(model.transitions[s][i], model.subs, model.version);
            if (w <= 0 || w > 1) return false;
            for (int k = 0; k < model.k; k++) {
                ws[k] = ws[k] + eval_entry(model.dtransitions[s][k][i], model.subs, model.version) / w;
            }
            s = model.transitionsSUM[s][i].id;
        }
        s = s - model.n;
        reward = model.rewards[s];
        for (int k = 0; k < model.k; k++) {
            g[k] = g[k] + reward * ws[k];   
        }

        avg_reward += reward;
    }

    for (int k = 0; k < model.k; k++) {
        g[k] /= model.nsample;
    }
    model.avg_reward = avg_reward / model.nsample;

    return optimize(model, g);
}

bool UniformPathUpdate(SparseMarkovModel& model) {
    int s, i, m;
    double reward, x;
    Weights ws(model.k, 0.0);
    Gradient g;
    
    g.resize(model.k, 0.0);

    double avg_reward = 0.0;
    for (int n = 0; n < model.nsample; n++) {
        fill(ws.begin(), ws.end(), 0.0);
        s = 0;
        while (s < model.n) {
            m = model.transitions[s].size();
            x = rng();
            i = int(m * x);

            for (int k = 0; k < model.k; k++) {
                ws[k] = ws[k] + m * eval_entry(model.dtransitions[s][k][i], model.subs, model.version);
            }
            s = model.transitionsSUM[s][i].id;
        }
        s = s - model.n;
        reward = model.rewards[s];
        for (int k = 0; k < model.k; k++) {
            g[k] = g[k] + reward * ws[k];   
        }

        avg_reward += reward;
    }

    for (int k = 0; k < model.k; k++) {
        g[k] /= model.nsample;
    }
    model.avg_reward = avg_reward / model.nsample;

    return optimize(model, g);
}

bool JacobiIterationUpdate(SparseMarkovModel& model) {
    // Build the derivated system
    vector<Eigen::Triplet<double>> triplets;
    int N = (model.k + 1) * model.nstate, j;
    double v;
    triplets.reserve((2 * model.k + 1) * model.ntrans + (model.k + 1) * model.nout); 

    for (int i = 0; i < model.n; i++) { // transitions
        for (size_t id_j = 0; id_j < model.transitions[i].size(); id_j++) {
            j = model.transitions[i][id_j].id;
            v = eval_expr(model.transitions[i][id_j].expr, model.subs);

            for (int k = 0; k < model.k; k++) {
                triplets.emplace_back(k * model.nstate + i, k * model.nstate + j, v);
                triplets.emplace_back(k * model.nstate + i, model.k * model.nstate + j, eval_expr(model.dtransitions[i][k][id_j].expr, model.subs));
            }
            triplets.emplace_back(model.k * model.nstate + i, model.k * model.nstate + j, v);
        }
    }
    for (int i = model.n; i < model.nstate; i++) {
        for (int k = 0; k < model.k; k++) {
            triplets.emplace_back(k * model.nstate + i, k * model.nstate + i, 1.0);
        }
        triplets.emplace_back(model.k * model.nstate + i, model.k * model.nstate + i, 1.0);
    }

    Eigen::SparseMatrix<double> A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // Build the output vector
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
    for (int i = 0; i < model.nout; i++)
        b[model.k * model.nstate + model.n + i] = model.rewards[i];

    // Iterate the output
    Eigen::VectorXd b_next(N);
    for (int i = 0; i < model.nsample; i++) {
        b_next.noalias() = A * b;
        b = b_next;
    }

    // Compute the gradient
    Gradient g(model.k);
    for (int k = 0; k < model.k; k++)
        g[k] = b[k * model.nstate];

    return optimize(model, g);
}

bool Update(SparseMarkovModel& model) {
switch (model.gradientmethod) {
    case GradientMethod::SparseLU:        return SparseLUUpdate(model);
    case GradientMethod::LogDerivative:   return LogDerivativeUpdate(model);
    case GradientMethod::UniformPath:     return UniformPathUpdate(model); 
    case GradientMethod::BiCGSTAB:        return BiCGSTABUpdate(model);
    case GradientMethod::JacobiIteration: return JacobiIterationUpdate(model);
    default:                              return false;
    }
}