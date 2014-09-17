/**
* NeuralLayer
*
* This class represents a neural network layer with a fixed size.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#ifndef _NEURAL_LAYER_H
#define _NEURAL_LAYER_H

using namespace std;

class NeuralLayer {
public:
	NeuralLayer();

	NeuralLayer(int numNeurons, int numInputsPerNeuron, bool hasBias = true);

	~NeuralLayer();

	/**
	* Number of neurons in this layer
	*/
	int numNeurons;

	/**
	* The neurons of the layer.
	*/
	vector<Neuron> neurons;

	/**
	* True if the layer has an additional bias value.
	*/
	bool hasBias;
};

#endif