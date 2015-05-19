/**
* Layer
*
* The implementation of a neural net layer.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#include "Layer.h"

#include <iostream>

using namespace std;

Layer::Layer(const size_t numNeurons, const size_t numInputsPerNeuron, const bool hasBias)
    : numNeurons(numNeurons),
      hasBias(hasBias)
{
    for (size_t n = 0; n < numNeurons; ++n) {
        this->neurons.push_back(Neuron(numInputsPerNeuron, hasBias));
    }
}

Layer::Type Layer::getType() const {
    return this->type;
}
