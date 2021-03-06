/**
* Neuron
*
* The Neuron class represents a single neuron with its weights to the neighbours.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#ifndef _NEURON_H
#define _NEURON_H

#include <iostream>
#include <vector>

using namespace std;

class Neuron {
public:
    Neuron();

    Neuron(size_t numInputs, bool hasBias = true);

    ~Neuron();

    /**
    * Weights of the neuron / synapse in biological terms
    */
    vector<double> weights;

    /**
    * The last changes of the weights - optimisation for the backpropagation algorithm
    */
    vector<double> deltaWeights;

    /**
    * The number of inputs of the neuron
    */
    size_t numInputs;

    /**
    * The sum of all the inputs
    */
    double netInput;
};

#endif
