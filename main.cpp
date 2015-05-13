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

#include <iostream>
#include <time.h>
#include <stdlib.h>

using namespace std;

int main() {
    srand(time(0));

    NeuralNet net(2, 1, 1, 2);

    net.setLearningRate(0.5);
    net.setMomentum(0.9);

    // Inputs

    vector<double> inp1, inp2, inp3, inp4;
    inp1.push_back(0);
    inp1.push_back(0);

    inp2.push_back(1);
    inp2.push_back(1);

    inp3.push_back(1);
    inp3.push_back(0);

    inp4.push_back(0);
    inp4.push_back(1);

    // Outputs

    vector<double> outputs;
    vector<double> out0, out1;
    out0.push_back(0);
    out1.push_back(1);

    double error = 0;
    for (int X = 0; X <= 15000; X++) {
        error = net.backpropagation(inp3, out1);
        error += net.backpropagation(inp2, out0);
        error += net.backpropagation(inp4, out1);
        error += net.backpropagation(inp1, out0);

        if (X % 1000 == 0) {
            error /= 4;
            cout << "Error = \t" << error << endl;
        }
    }

    if(!net.save("export/nn.json"))
        cout << "Could not export file" << endl;

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

    return 0;
}
