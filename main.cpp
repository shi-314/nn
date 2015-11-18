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
#include <time.h>
#include <stdlib.h>

using namespace std;

int main() {
    srand(time(0));

    NeuralNet net("xornet");

    net.add(Layer::INPUT, 2);
    net.add(Layer::HIDDEN, 2);
    net.add(Layer::OUTPUT, 1);

    net.setLearningRate(0.5);
    net.setMomentum(0.9);

    // Inputs

    vector<double> inp1 = {0, 0},
                   inp2 = {1, 1},
                   inp3 = {1, 0},
                   inp4 = {0, 1};

    // Outputs

    vector<double> outputs;
    vector<double> out0 = {0}, out1 = {1};

    double error = 0;

    auto start = chrono::high_resolution_clock::now();
    int i;
    for (i = 0; i < 100000; i++) {
        error = net.backpropagation(inp3, out1);
        error += net.backpropagation(inp2, out0);
        error += net.backpropagation(inp4, out1);
        error += net.backpropagation(inp1, out0);

        if (i % 10000 == 0) {
            error /= 4;
            cout << "Error = \t" << error << endl;
        }
    }

    cout << endl;

    auto end = chrono::high_resolution_clock::now();
    cout << "Backpropagation: " << i << " iterations in ";
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms\n";

    cout << endl;

    // Let's test the neural network :)

    cout << "Testing the neural network...\n";

    cout << "Input: 1,1\n";
    net.calculateOutputs(inp2);
    outputs = net.getOutputs();
    cout << "Output: " << outputs[0] << endl;

    cout << "Input: 0,0\n";
    net.calculateOutputs(inp1);
    outputs = net.getOutputs();
    cout << "Output: " << outputs[0] << endl;

    cout << "Input: 1,0\n";
    net.calculateOutputs(inp3);
    outputs = net.getOutputs();
    cout << "Output: " << outputs[0] << endl;

    cout << "Input: 0,1\n";
    net.calculateOutputs(inp4);
    outputs = net.getOutputs();
    cout << "Output: " << outputs[0] << endl;

    cout << endl;

    if (!net.save("export/nn.json"))
        cout << "Could not export file" << endl;

    return 0;
}
