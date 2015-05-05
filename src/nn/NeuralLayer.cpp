/**
* NeuralLayer
*
* The implementation of a neural net layer.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#include "NeuralLayer.h"

NeuralLayer::NeuralLayer()
    : numNeurons(0),
      hasBias(true)
{
}

NeuralLayer::NeuralLayer(const size_t numNeurons, const size_t numInputsPerNeuron, const bool hasBias)
    : numNeurons(numNeurons),
      hasBias(hasBias)
{
    for (size_t n = 0; n < numNeurons; ++n) {
        this->neurons.push_back(Neuron(numInputsPerNeuron, hasBias));
    }
}
