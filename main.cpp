/**
* Neural Net
*
* This is example shows the usage of nn by creating a neural network that learns the XOR function
* with the backpropagation algorithm.
*
* @author Shivan Taher
* @date 28.04.2009
*/

#include <nn/NeuralNet.h>

#include <chrono>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>

using namespace std;

int main() {
    srand(time(0));

    NeuralNet net("xornet");

    net.add(Layer::INPUT, 2);
    net.add(Layer::HIDDEN, 2);
    net.add(Layer::OUTPUT, 1);

    auto softmax = [] (double x) {
        return 1 / (1 + exp(-x));
    };

    auto softmaxDerivative = [] (double x) {
        return 1.0 / (1.0 + exp(-x));
    };

    auto hardmax = [] (double x) {
        return max(0.0, x);
    };

    auto hardmaxDerivative = [] (double x) {
        return x > 0 ? 1.0 : 0.0;
    };

    net.setActivationFunction(hardmax, hardmaxDerivative);

    net.setBiasValue(0.5);
    net.setLearningRate(0.1);
    net.setMomentum(0.9);

    // Inputs

    typedef vector<double> InputType;
    typedef vector<double> OutputType;
    typedef pair<InputType, OutputType> DataSet;

    vector<DataSet> data = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 1}, {0}},
        {{1, 0}, {1}}
    };

    cout << "Learning..." << endl;

    vector<double> outputs;
    double error = 0;

    int i;
    size_t d;

    auto start = std::chrono::high_resolution_clock::now();
    for (i = 0; i < 1; i++) {
        error = 0;
        for (d = 0; d < data.size(); ++d) {
            net.calculateOutputs(data[d].first);
            error += net.backpropagation(data[d].first, data[d].second);
        }

        if (i % 10000 == 0) {
            net.setLearningRate(net.getLearningRate() * 0.1);
            error /= data.size();
            cout << "\terror = " << error << endl;
        }
    }

    cout << endl;

    auto end = std::chrono::high_resolution_clock::now();
    cout << i << " iterations in ";
    cout << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    cout << endl;

    // Let's test the neural network :)

    cout << "Testing the neural network...\n";

    for (d = 0; d < data.size(); ++d) {
        net.calculateOutputs(data[d].first);
        outputs = net.getOutputs();

        cout << "\tf( ";
        for(auto i : data[d].first) {
            cout << i << " ";
        }
        cout << ") = ";
        for(auto o : outputs) {
            cout << o << " ";
        }
        cout << endl;
    }

    cout << endl;

    if (!net.save("export/nn.json"))
        cout << "Could not export file" << endl;

    return 0;
}
