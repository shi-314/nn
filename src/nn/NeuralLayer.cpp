/**
* NeuralLayer
*
* The implementation of a neural net layer.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#include "NeuralNetModule.h"

// ============================================================= //
// Konstruktor & Destruktor
// ============================================================= //
NeuralLayer::NeuralLayer() {
	this->numNeurons = 0;
}

NeuralLayer::NeuralLayer(int numNeurons, int numInputsPerNeuron, bool hasBias) {
	this->hasBias = hasBias;
	this->numNeurons = numNeurons;
	for (int n = 0; n < numNeurons; ++n) {
		this->neurons.push_back(Neuron(numInputsPerNeuron, hasBias));
	}
}

NeuralLayer::~NeuralLayer() {

}