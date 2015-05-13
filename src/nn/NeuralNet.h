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

#include <iostream>
#include <vector>

#include "NeuralLayer.h"

using namespace std;

class NeuralNet {
public:
    NeuralNet();
    NeuralNet(size_t inputs, size_t outputs, size_t hiddenLayers,
        size_t neuronsPerHL);

    /**
    * Sets the number of input units
    */
    void setNumInputs(size_t n);

    /**
    * Returns the number of input units
    */
    size_t getNumInputs() const;

    /**
    * Sets the number of output units
    */
    void setNumOutputs(size_t n);

    /**
    * Returns the number of output units
    */
    size_t getNumOutputs() const;

    /**
    * Sets the number of hidden layers
    */
    void setNumHiddenLayers(size_t n);

    /**
    * Returns the number of hidden layers
    */
    size_t getNumHiddenLayers() const;

    /**
    * Sets the number of neurons in each hidden layer
    */
    void setNumNeuronsPerHL(size_t n);

    /**
    * Returns the number of neurons in each hidden layer
    */
    size_t getNumNeuronsPerHL() const;

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
    void setLearningRate(double value);

    /**
    * Returns the learning rate of the backpropagation algorithm.
    */
    double getLearningRate() const;

    /**
    * Sets the momentum value (Trägheitsterm)
    */
    void setMomentum(double value);

    /**
    * Returns the momentum value (Trägheitsterm)
    */
    double getMomentum() const;

    /**
    * Sets the bias values - 0 ignores the bias
    */
    void setBiasValue(double bias);

    /**
    * Returns the bias value
    */
    double getBiasValue() const;

    /**
    * Enables or disables the bias
    */
    void setBiasStatus(bool useBias);

    /**
    * Returns true if the bias value is enabled
    */
    bool getBiasStatus() const;

    /**
    * Saves the neural network in an XML file
    */
    bool saveFile(const string& filename);
    bool save(const string& filename);

    /**
    * Loads the neural network from an XML file
    */
    bool loadFile(const string& filename);

    /**
    * Sigmoid function (activation function)
    */
    inline double sigmoid(double x);

    /**
    * The first derivation of the sigmoid function
    */
    inline double sigmoidDerivation(double x);

private:
    size_t numInputs;
    size_t numOutputs;
    size_t numHiddenLayers;
    size_t numNeuronsPerHL;

    double momentum;
    double learningRate;
    double biasValue;
    bool useBias;

    vector<NeuralLayer> layers;
    vector<double> outputs;
};

#endif
