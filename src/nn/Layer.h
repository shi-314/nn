/**
* Layer
*
* This class represents a neural network layer with a fixed size.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#ifndef _NEURAL_LAYER_H
#define _NEURAL_LAYER_H

#include <iostream>
#include <vector>

#include "Neuron.h"

using namespace std;

class Layer {
public:
    enum Type { INPUT, HIDDEN, OUTPUT };

    Layer(const size_t numNeurons, const size_t numInputsPerNeuron, const bool hasBias = true);
    ~Layer();

    /**
    * Number of neurons in this layer
    */
    size_t numNeurons;

    /**
    * The neurons of the layer.
    */
    vector<Neuron*> neurons;

    /**
    * True if the layer has an additional bias value.
    */
    bool hasBias;

    Type getType() const;
    
private:
    Type type;
};

#endif
