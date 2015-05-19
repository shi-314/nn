/**
* NeuralNetwork
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

#include "Layer.h"

using namespace std;

class NeuralNet {
public:
    NeuralNet(const string& name);
    ~NeuralNet();

    void add(Layer::Type layerType, size_t numNeurons);

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
    * Saves the neural network as a JSON file
    */
    bool save(const string& filename);

    /**
    * Loads the neural network from a JSON file
    */
    bool load(const string& filename);

    /**
    * Sigmoid function (activation function)
    */
    inline double sigmoid(double x);

    /**
    * The first derivation of the sigmoid function
    */
    inline double sigmoidDerivation(double x);

private:
    size_t numOutputs;
    size_t numHiddenLayers;

    double momentum;
    double learningRate;
    double biasValue;
    bool useBias;

    vector<Layer*> layers;
    vector<double> outputs;
    string name;
};

#endif
