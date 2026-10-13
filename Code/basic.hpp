#pragma once

#include <symengine/expression.h>
#include <symengine/derivative.h>
#include <symengine/eval_double.h>
#include <symengine/dict.h>
#include <iostream>
#include <iomanip> 
#include <stdio.h>
#include <random>
#include <chrono>
#include <cmath>
#include <Eigen/Sparse>

using namespace std;
using SymEngine::Expression;
using SymEngine::eval_double;
using SymEngine::map_basic_basic;
using std::chrono::duration;

struct Entry {
    int id;
    Expression expr;
    mutable int version;
    mutable double val;

    Entry(int id, Expression expr) : id(id), expr(expr), version(-1), val(0.0) {}
};

using SparseVector = vector<Entry>;
using SparseMatrix = vector<SparseVector>;
using SparseTensor = vector<SparseMatrix>;
using Variables = vector<Expression>;
using Rewards = vector<double>;
using Vector = vector<Expression>;
using Matrix = vector<Vector>;
using Point = vector<double>;
using Gradient = Point;
using Weights = vector<double>;
using Interval = pair<double, double>;

map_basic_basic create_subs(const Variables& vars, const Point& p) {
    map_basic_basic subs;
    for (int i = 0; i < int(p.size()); i++) {
        subs[vars[i].get_basic()] = Expression(p[i]).get_basic();
    }

    return subs;
}

double eval_expr(const Expression& e, const map_basic_basic& subs) {
    Expression result = e.subs(subs);
    double evaluated = eval_double(*result.get_basic());
    return evaluated;
}

double eval_entry(const Entry& x, const map_basic_basic& subs, int version) {
    if (x.version != version) {
        x.val = eval_expr(x.expr, subs);
        x.version = version;
    }

    return x.val;
}

enum class OptimizerType {SGD, SmallSGD, AdaGrad, Adam, AdamW, CAdamW, CAdaGrad};
enum class GradientMethod {LogDerivative, SparseLU, UniformPath, BiCGSTAB, JacobiIteration};

mt19937 gen(random_device{}());
uniform_real_distribution<double> dist(0.0, 1.0);

double rng() {
    return dist(gen);
}

struct SparseMarkovModel {
    const int n, k, nout, nloop, nsample, ntrans, nstate;
    const OptimizerType optimizer;
    const GradientMethod gradientmethod;
    const double timeout, lr, lr_decay, weight_decay, alpha, beta, eps;
    double avg_reward;
    const Variables vars;
    const vector<Interval> intervals;
    const SparseMatrix transitions, transitionsSUM;
    const SparseTensor dtransitions;
    const Rewards rewards;
    Point p;
    map_basic_basic subs;
    Gradient m, v;
    int version;

    SparseMarkovModel(int n, int k, int nloop, double timeout, int nsample, int ntrans, OptimizerType optimizer, GradientMethod gradientmethod, double lr, double lr_decay, double weight_decay, double alpha, double beta, double eps, Variables& vars, vector<Interval>& intervals, SparseMatrix& transitions, SparseMatrix& transitionsSUM, SparseTensor& dtransitions, Rewards& rewards, Point& p)
        : n(n), k(k), nout(rewards.size()), nloop(nloop), nsample(nsample), ntrans(ntrans), nstate(n + rewards.size()),
        optimizer(optimizer), gradientmethod(gradientmethod), timeout(timeout), lr(lr), lr_decay(lr_decay), weight_decay(weight_decay), alpha(alpha), beta(beta), eps(eps), 
        avg_reward(0.0), vars(vars), intervals(intervals),
        transitions(transitions), transitionsSUM(transitionsSUM), dtransitions(dtransitions), 
        rewards(rewards), p(p), subs(create_subs(vars, p)), m(k, 0.0), v(k, 0.0), version(0) 
        {}
};

double get_score(const SparseMarkovModel& model, const Point& p) {
    // 1. Build (I - A)
    vector<Eigen::Triplet<double>> triplets;
    int N = model.n + model.nout;
    triplets.reserve(model.ntrans + N);

    for (int i = 0; i < N; i++) // I
        triplets.emplace_back(i, i, 1.0);

    map_basic_basic subs = create_subs(model.vars, p);
    for (int i = 0; i < model.n; i++) // -transitions
        for (const auto& e : model.transitions[i])
            triplets.emplace_back(i, e.id, -eval_expr(e.expr, subs));

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

    return x[0];
}