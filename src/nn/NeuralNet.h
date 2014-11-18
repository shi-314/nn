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
	NeuralNet(const size_t inputs, const size_t outputs, const size_t hiddenLayers, const size_t neuronsPerHL);

	~NeuralNet();

	/**
	* Sets the number of input units
	*/
	void setNumInputs(const size_t n);

	/**
	* Returns the number of input units
	*/
	int getNumInputs() const;

	/**
	* Sets the number of output units
	*/
	void setNumOutputs(const size_t n);

	/**
	* Returns the number of output units
	*/
	int getNumOutputs() const;

	/**
	* Sets the number of hidden layers
	*/
	void setNumHiddenLayers(const size_t n);

	/**
	* Returns the number of hidden layers
	*/
	int getNumHiddenLayers() const;

	/**
	* Sets the number of neurons in each hidden layer
	*/
	void setNumNeuronsPerHL(const size_t n);

	/**
	* Returns the number of neurons in each hidden layer
	*/
	int getNumNeuronsPerHL() const;

	/**
	* Creates the neural network with the previously defined dimensions
	*/
	void createNet();

	/**
	* Sends the signals (inputs) through the neural network und
	* returns the calculated output values.
	*/
	const vector<double>& calculateOutputs(vector<double> inputs);

	/**
	* Returns the last output values
	*/
	const vector<double>& getOutputs() const;

	/**
	* Applies the backpropagation algorithm to the neural network and returns the standard error.
	*/
	double backpropagation(const vector<double>& inputs, const vector<double>& expectedOutputs);

	/**
	* Sets the learning rate for the backpropagation algorithm.
	*/
	void setLearningRate(const double value);

	/**
	* Returns the learning rate of the backpropagation algorithm.
	*/
	double getLearningRate() const;

	/**
	* Sets the momentum value (Trägheitsterm)
	*/
	void setMomentum(const double value);

	/**
	* Returns the momentum value (Trägheitsterm)
	*/
	double getMomentum() const;

	/**
	* Sets the bias values - 0 ignores the bias
	*/
	void setBiasValue(const double bias);

	/**
	* Returns the bias value
	*/
	double getBiasValue() const;

	/**
	* Enables or disables the bias
	*/
	void setBiasStatus(const bool useBias);

	/**
	* Returns true if the bias value is enabled
	*/
	bool getBiasStatus() const;

	/**
	* Saves the neural network in an XML file
	*/
	bool saveFile(const string& filename);

	/**
	* Loads the neural network from an XML file
	*/
	bool loadFile(const string& filename);

	/**
	* Sigmoid function (activation function)
	*/
	inline double sigmoid(const double& x);

	/**
	* The first derivation of the sigmoid function
	*/
	inline double sigmoidDerivation(const double& x);

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