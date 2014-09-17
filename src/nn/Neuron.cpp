/**
* Neuron
*
* The Neuron class represents a single neuron with its weights to the neighbours.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#include "NeuralNetModule.h"

Neuron::Neuron() {
	this->netInput = 0;
	this->numInputs = 0;
}

Neuron::Neuron(int numInputs, bool hasBias) {
	this->netInput = 0;
	this->numInputs = numInputs;

	if (hasBias) {
		this->numInputs += 1;
	}

	double XMin = -1;
	double XMax = 1;

	for (int n = 0; n < this->numInputs; ++n) {
		double x = randomDouble(XMin, XMax);
		this->weights.push_back(x);
		this->deltaWeights.push_back(0);
	}
}

Neuron::~Neuron() {
}