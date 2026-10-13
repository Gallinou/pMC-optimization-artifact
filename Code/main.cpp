#include "parser.hpp"
#include "gradient.hpp"

int main() {
    std::cout << std::defaultfloat << std::setprecision(std::numeric_limits<double>::max_digits10);

    auto t0 = std::chrono::steady_clock::now();
    SparseMarkovModel model = GenerateSparceMarkovModel();

    if (model.nloop == 0) std::cout << "No optimization step limit." << endl;
    else if (model.nloop == 1) std::cout << "First gradient only." << endl;
    else std::cout << "Max number of steps: " << model.nloop << endl;

    if (model.timeout == 0.0) std::cout << "No optimization time limit." << endl;
    else std:: cout << "Max time: " << model.timeout << endl;

    switch (model.gradientmethod) {
        case GradientMethod::SparseLU:        std::cout << "Gradient method: SparseLU" << endl;        break;
        case GradientMethod::LogDerivative:   std::cout << "Gradient method: LogDerivative" << endl;   break;
        case GradientMethod::UniformPath:     std::cout << "Gradient method: UniformPath" << endl;     break;    
        case GradientMethod::BiCGSTAB:        std::cout << "Gradient method: BiCGSTAB" << endl;        break;
        case GradientMethod::JacobiIteration: std::cout << "Gradient method: JacobiIteration" << endl; break;
        default:                                                                                       break;
    }

    double chrono = 0.0;
    std::cout << "[ ";
    for (int k = 0; k < model.k; k++) std::cout << model.vars[k] << "=" << model.p[k] << " ";
    if (model.nloop == 1)
        std::cout << "] time=" << chrono << endl;
    else {
        double score = get_score(model, model.p);
        std::cout << "] score=" << score << " time=" << chrono << endl;
    }
    
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Time for model initialization: " << dt.count() << endl;

    int step = 0;
    while (true) {
        t0 = std::chrono::steady_clock::now();
        if (!Update(model))
            break;
        t1 = std::chrono::steady_clock::now();
        dt = t1 - t0;
        chrono += dt.count();

        if (model.timeout != 0.0)
            if (chrono > model.timeout)
                break;

        std::cout << "[ ";
        for (int k = 0; k < model.k; k++) std::cout << model.vars[k] << "=" << model.p[k] << " ";
        if (model.nloop == 1)
            std::cout << "] time=" << chrono << endl;
        else {
            double score = get_score(model, model.p);
            std::cout << "] score=" << score << " time=" << chrono << endl;
        } 

        step++;
        if (model.nloop != 0)
            if (step == model.nloop)
                break;
    }

    return 0;
}