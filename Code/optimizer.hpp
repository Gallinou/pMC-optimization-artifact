#include "basic.hpp"

double opt_prec = 0.0;

bool SGD(SparseMarkovModel& model, const Gradient& g) {
    double new_pk;
    Point new_p;
    new_p.reserve(model.k);

    for (int k = 0; k < model.k; k++) {
        new_pk = model.p[k] + model.eps * g[k];

        if (new_pk < model.intervals[k].first + opt_prec || model.intervals[k].second - opt_prec < new_pk)
            return false;

        new_p.emplace_back(new_pk);
    }

    model.p = std::move(new_p);
    model.subs = create_subs(model.vars, model.p);
    return true;
}

bool smallSGD(SparseMarkovModel& model, const Gradient& g) {
    double mineps = model.eps * 1e-2;
    double alpha = model.eps;

    for (int k = 0; k < model.k; k++) {
        if (g[k] > 0) {
            alpha = min(alpha, ((model.intervals[k].second - opt_prec) - model.p[k]) / g[k]);
        }
        else if (g[k] < 0) {
            alpha = min(alpha, (model.p[k] - (model.intervals[k].first + opt_prec)) / -g[k]);
        }
    }

    if (alpha < mineps)
        return false;

    double new_pk;
    Point new_p;
    new_p.reserve(model.k);

    for (int k = 0; k < model.k; k++) {
        new_pk = model.p[k] + alpha * g[k];

        if (new_pk < model.intervals[k].first + opt_prec || model.intervals[k].second - opt_prec < new_pk)
            return false;

        new_p.emplace_back(new_pk);
    }

    model.p = std::move(new_p);
    model.subs = create_subs(model.vars, model.p);
    return true;
}

bool AdaGrad(SparseMarkovModel& model, const Gradient& g) {
    double new_pk, gk;
    Point new_p;
    new_p.reserve(model.k);
    
    double current_lr = model.lr / (1 + (model.version - 1) * model.lr_decay);

    for (int k = 0; k < model.k; k++) {
        gk = g[k] + model.weight_decay * model.p[k];
        model.m[k] += gk * gk;
        new_pk = model.p[k] + current_lr * gk / (sqrt(model.m[k]) + model.eps);

        if (new_pk < model.intervals[k].first + opt_prec || model.intervals[k].second - opt_prec < new_pk)
            return false;

        new_p.emplace_back(new_pk);
    }

    model.p = std::move(new_p);
    model.subs = create_subs(model.vars, model.p);
    return true;
}

bool Adam(SparseMarkovModel& model, const Gradient& g) {
    double new_pk, gk, mk, vk;
    Point new_p;
    new_p.reserve(model.k);

    for (int k = 0; k < model.k; k++) {
        gk = g[k] + model.weight_decay * model.p[k];

        model.m[k] = model.alpha * model.m[k] + (1 - model.alpha) * gk;
        model.v[k] = model.beta * model.v[k] + (1 - model.beta) * gk * gk;

        mk = model.m[k] / (1 - pow(model.alpha, model.version));
        vk = model.v[k] / (1 - pow(model.beta, model.version));

        new_pk = model.p[k] + model.lr * mk / (sqrt(vk) + model.eps);

        if (new_pk < model.intervals[k].first + opt_prec || model.intervals[k].second - opt_prec < new_pk)
            return false;

        new_p.emplace_back(new_pk);
    }

    model.p = std::move(new_p);
    model.subs = create_subs(model.vars, model.p);
    return true;
}

bool AdamW(SparseMarkovModel& model, const Gradient& g) {
    double new_pk, gk, mk, vk;
    Point new_p;
    new_p.reserve(model.k);

    for (int k = 0; k < model.k; k++) {
        gk = g[k];
        new_pk = (1 - model.lr * model.weight_decay) * model.p[k];

        model.m[k] = model.alpha * model.m[k] + (1 - model.alpha) * gk;
        model.v[k] = model.beta * model.v[k] + (1 - model.beta) * gk * gk;

        mk = model.m[k] / (1 - pow(model.alpha, model.version));
        vk = model.v[k] / (1 - pow(model.beta, model.version));

        new_pk += model.lr * mk / (sqrt(vk) + model.eps);

        if (new_pk < model.intervals[k].first + opt_prec || model.intervals[k].second - opt_prec < new_pk)
            return false;

        new_p.emplace_back(new_pk);
    }

    model.p = std::move(new_p);
    model.subs = create_subs(model.vars, model.p);
    return true;
}

bool CAdamW(SparseMarkovModel& model, const Gradient& g) {
    double new_pk, gk, mk, vk, a, b;
    Point new_p;
    new_p.reserve(model.k);

    for (int k = 0; k < model.k; k++) {
        gk = g[k];
        new_pk = (1 - model.lr * model.weight_decay) * model.p[k];

        model.m[k] = model.alpha * model.m[k] + (1 - model.alpha) * gk;
        model.v[k] = model.beta * model.v[k] + (1 - model.beta) * gk * gk;

        mk = model.m[k] / (1 - pow(model.alpha, model.version));
        vk = model.v[k] / (1 - pow(model.beta, model.version));

        new_pk += model.lr * mk / (sqrt(vk) + model.eps);

        a = model.intervals[k].first;
        b = model.intervals[k].second;

        if (new_pk < a + opt_prec) 
            new_pk = a;
        else if (b - opt_prec < new_pk)
            new_pk = b;

        new_p.emplace_back(new_pk);
    }

    model.p = std::move(new_p);
    model.subs = create_subs(model.vars, model.p);
    return true;
}

bool CAdaGrad(SparseMarkovModel& model, const Gradient& g) {
    double new_pk, gk, a, b;
    Point new_p;
    new_p.reserve(model.k);
    
    double current_lr = model.lr / (1 + (model.version - 1) * model.lr_decay);

    for (int k = 0; k < model.k; k++) {
        gk = g[k] + model.weight_decay * model.p[k];
        model.m[k] += gk * gk;
        new_pk = model.p[k] + current_lr * gk / (sqrt(model.m[k]) + model.eps);

        a = model.intervals[k].first;
        b = model.intervals[k].second;

        if (new_pk < a + opt_prec) 
            new_pk = a;
        else if (b - opt_prec < new_pk)
            new_pk = b;

        new_p.emplace_back(new_pk);
    }

    model.p = std::move(new_p);
    model.subs = create_subs(model.vars, model.p);
    return true;
}

bool optimize(SparseMarkovModel& model, const Gradient& g) {
    if (model.version == 0) {
        cout << "First gradient: ";
        for (int k = 0; k < model.k; k++)
            cout << g[k] << " ";
        cout << endl;
    }

    if (model.nloop == 1) return true;

    model.version++;
    switch (model.optimizer)
    {
    case OptimizerType::SGD:
        return SGD(model, g);

    case OptimizerType::SmallSGD:
        return smallSGD(model, g);
    
    case OptimizerType::AdaGrad:
        return AdaGrad(model, g);
    
    case OptimizerType::Adam:
        return Adam(model, g);

    case OptimizerType::AdamW:
        return AdamW(model, g);

    case OptimizerType::CAdamW:
        return CAdamW(model, g);

    case OptimizerType::CAdaGrad:
        return CAdaGrad(model, g);

    default:
        return false;
    }
}