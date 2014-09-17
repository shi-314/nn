/**
* NeuralLayer
*
* This class represents a neural network layer with a fixed size.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H

class NeuralNet {
public:
	NeuralNet();

	~NeuralNet();

	/**
	* Sets the number of input units
	*/
	void setNumInputs(int n);

	/**
	* Returns the number of input units
	*/
	int getNumInputs();

	/**
	* Sets the number of output units
	*/
	void setNumOutputs(int n);

	/**
	* Returns the number of output units
	*/
	int getNumOutputs();

	/**
	* Sets the number of hidden layers
	*/
	void setNumHiddenLayers(int n);

	/**
	* Returns the number of hidden layers
	*/
	int getNumHiddenLayers();

	/**
	* Sets the number of neurons in each hidden layer
	*/
	void setNumNeuronsPerHL(int n);

	/**
	* Returns the number of neurons in each hidden layer
	*/
	int getNumNeuronsPerHL();

	/**
	* Creates the neural network with the previously defined dimensions
	*/
	void createNet();

	/**
	* Sends the signals (inputs) through the neural network und
	* returns the calculated output values.
	*/
	vector<double> calculateOutputs(vector<double> inputs);

	/**
	* Returns the last output values
	*/
	vector<double> getOutputs();

	/**
	* Applies the backpropagation algorithm to the neural network and returns the standard error.
	*/
	double backpropagation(vector<double> inputs, vector<double> expectedOutputs);

	/**
	* Sets the learning rate for the backpropagation algorithm.
	*/
	void setLearningRate(double value);

	/**
	* Returns the learning rate of the backpropagation algorithm.
	*/
	double getLearningRate();

	/**
	* Sets the momentum value (Trägheitsterm)
	*/
	void setMomentum(double value);

	/**
	* Returns the momentum value (Trägheitsterm)
	*/
	double getMomentum();

	/**
	* Sets the bias values - 0 ignores the bias
	*/
	void setBiasValue(double bias);

	/**
	* Returns the bias value
	*/
	double getBiasValue();

	/**
	* Enables or disables the bias
	*/
	void setBiasStatus(bool useBias);

	/**
	* Returns true if the bias value is enabled
	*/
	bool getBiasStatus();

	/**
	* Saves the neural network in an XML file
	*/
	bool saveFile(string filename);

	/**
	* Loads the neural network from an XML file
	*/
	bool loadFile(string filename);

	/**
	* Sigmoid function (activation function)
	*/
	inline double sigmoid(double x);

	/**
	* The first derivation of the sigmoid function
	*/
	inline double sigmoidDerivation(double x);

private:
	int numInputs;
	int numOutputs;
	int numHiddenLayers;
	int numNeuronsPerHL;

	double momentum;
	double learningRate;
	double biasValue;
	bool useBias;

	vector<NeuralLayer> layers;
	vector<double> outputs;
};

#endif