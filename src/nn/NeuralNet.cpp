/**
* NeuralNet
*
* This class can be used to create a neural network with a given size and provides the backpropagation algorithm and
* funtions to save and load a neural network from a json file.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#include "NeuralNet.h"
#include "Utils.h"

#include <json/json.h>

#include <fstream>
#include <math.h>

NeuralNet::NeuralNet()
    : numInputs(0),
      numOutputs(0),
      numHiddenLayers(0),
      numNeuronsPerHL(0),
      momentum(0.9),
      learningRate(1),
      biasValue(1),
      useBias(true),
      name("")
{
}

void NeuralNet::add(Layer& layer) {
    this->layers.push_back(layer);
}

NeuralNet::NeuralNet(size_t inputs, size_t outputs, size_t hiddenLayers,
    size_t neuronsPerHL)
    : numInputs(inputs),
      numOutputs(outputs),
      numHiddenLayers(hiddenLayers),
      numNeuronsPerHL(neuronsPerHL),
      momentum(0.9),
      learningRate(1),
      biasValue(1),
      useBias(true)
{
    this->createNet();
}

void NeuralNet::setNumInputs(size_t n) {
    this->numInputs = n;
}

size_t NeuralNet::getNumInputs() const {
    return this->numInputs;
}

void NeuralNet::setNumOutputs(size_t n) {
    this->numOutputs = n;
}

size_t NeuralNet::getNumOutputs() const {
    return this->numOutputs;
}

void NeuralNet::setNumHiddenLayers(size_t n) {
    this->numHiddenLayers = n;
}

size_t NeuralNet::getNumHiddenLayers() const {
    return this->numHiddenLayers;
}

void NeuralNet::setNumNeuronsPerHL(size_t n) {
    this->numNeuronsPerHL = n;
}

size_t NeuralNet::getNumNeuronsPerHL() const {
    return this->numNeuronsPerHL;
}

void NeuralNet::createNet() {
    // Create the input layer
    Layer inputLayer(this->numInputs, 0, false); // Ein Input-Neuron hat selber keine Inputs
    this->layers.push_back(inputLayer);

    // Create the hidden layers
    for (size_t n = 0; n < this->numHiddenLayers; ++n) {
        if (n == 0) {
            Layer hiddenLayer(this->numNeuronsPerHL, this->numInputs, this->useBias);
            this->layers.push_back(hiddenLayer);
        } else {
            Layer hiddenLayer(this->numNeuronsPerHL, this->numNeuronsPerHL, this->useBias);
            this->layers.push_back(hiddenLayer);
        }
    }

    // Create the output layer
    Layer outputLayer(this->numOutputs, this->numNeuronsPerHL, this->useBias);
    this->layers.push_back(outputLayer);
}

double NeuralNet::sigmoid(double x) {
    //double response = 1;
    //return 1/(1+exp(-x/response));
    return 1 / (1 + exp(-x));
}

double NeuralNet::sigmoidDerivation(double x) {
    double gx = this->sigmoid(x);
    return gx * (1 - gx);
}

const vector<double>& NeuralNet::calculateOutputs(vector<double> inputs) {
    // Results
    this->outputs.clear();

    // Check the size of the inputs
    if (inputs.size() != this->numInputs)
        return outputs;

    // For each layer
    for (size_t i = 0; i < this->layers.size(); ++i) {
        if (i > 0) // The input is the output of the last layer
            inputs = this->outputs;

        this->outputs.clear();

        // Calculate the outputs = sigmoid(sum of (inputs * weights))

        for (size_t j = 0; j < this->layers[i].numNeurons; ++j) {
            double netinput = 0;

            size_t numInputs = this->layers[i].neurons[j].numInputs;

            // For each weight
            // Calculate the net input (sum of inputs * weights)

            // Ignore the input layer
            if (i > 0) {
                for (size_t k = 0; k < inputs.size(); ++k)
                    netinput += this->layers[i].neurons[j].weights[k] * inputs[k];
            } else {
                netinput = inputs[j];
            }

            // Add the bias value if enabled
            if (this->useBias && this->layers[i].hasBias) {
                netinput += this->layers[i].neurons[j].weights[numInputs - 1] * this->biasValue;
            }

            this->layers[i].neurons[j].netInput = netinput;
            this->outputs.push_back(sigmoid(netinput));
        }
    }
    return this->outputs;
}

const vector<double>& NeuralNet::getOutputs() const {
    return this->outputs;
}

double NeuralNet::backpropagation(const vector<double>& inputs, const vector<double>& expectedOutputs) {
    // Error values
    vector<double> delta_j, delta_i;
    double standardError = 0;

    // Calculate the activity of the network first
    this->calculateOutputs(inputs);

    //
    // Calculate and correct the errors of the output unit
    //

    for (size_t i = 0; i < this->numOutputs; i++) {
        // Err = y - aj = y - g(net_j)
        double err = expectedOutputs[i] - this->outputs[i];
        standardError += err;

        // netinput of the neuron i
        double net_i = this->layers[this->numHiddenLayers + 1].neurons[i].netInput;

        double di = err * this->sigmoidDerivation(net_i);
        delta_i.push_back(di);

        // Correct the weights between the output layer and the hidden layer

        size_t numInputsI = this->layers[this->numHiddenLayers + 1].neurons[i].numInputs;
        for (size_t j = 0; j < numInputsI; j++) {
            double net_j;

            // Does the layer have a bias and is j the bias neuron?
            if (this->layers[this->numHiddenLayers + 1].hasBias && j == numInputsI - 1) {
                net_j = this->biasValue;
            } else {
                net_j = this->layers[this->numHiddenLayers].neurons[j].netInput;
            }

            double delta_w = this->learningRate * this->sigmoid(net_j) * di + this->momentum * this->layers[this->numHiddenLayers + 1].neurons[i].deltaWeights[j];

            this->layers[this->numHiddenLayers + 1].neurons[i].deltaWeights[j] = delta_w;
            this->layers[this->numHiddenLayers + 1].neurons[i].weights[j] += delta_w;
        }
    }

    standardError /= this->numOutputs;
    standardError = (standardError * standardError) / 2; // E = 1/2 Err^2

    //
    // Calculate and correct the errors of the hidden layers
    //

    for (size_t L = this->numHiddenLayers; L > 0; L--) {
        for (size_t j = 0; j < this->numNeuronsPerHL; j++) {
            double err_j = 0;

            // Calculate the errors of the neuron j
            for (size_t i = 0; i < this->layers[L + 1].numNeurons; i++) {
                err_j += this->layers[L + 1].neurons[i].weights[j] * delta_i[i];
            }

            double net_j = this->layers[L].neurons[j].netInput;
            double dj = this->sigmoidDerivation(net_j) * err_j;
            delta_j.push_back(dj);

            // Correct the weights between the hidden layer and the predecessor layer

            size_t numInputsJ = this->layers[L].neurons[j].numInputs;
            for (size_t k = 0; k < numInputsJ; k++) {
                double net_k;

                // Does the layer have a bias and is j the bias neuron?
                if (this->layers[L].hasBias && k == numInputsJ - 1) {
                    net_k = this->biasValue;
                } else {
                    net_k = this->layers[L - 1].neurons[k].netInput;
                }
                double delta_w = this->learningRate * this->sigmoid(net_k) * dj + this->momentum * this->layers[L].neurons[j].deltaWeights[k];

                this->layers[L].neurons[j].deltaWeights[k] = delta_w;
                this->layers[L].neurons[j].weights[k] += delta_w;
            }
        }
        delta_i = delta_j;
        delta_j.clear();
    }

    return standardError;
}

void NeuralNet::setLearningRate(double value) {
    this->learningRate = value;
}

double NeuralNet::getLearningRate() const {
    return this->learningRate;
}

void NeuralNet::setMomentum(double value) {
    this->momentum = value;
}

double NeuralNet::getMomentum() const {
    return this->momentum;
}

void NeuralNet::setBiasValue(double bias) {
    this->biasValue = bias;
}

double NeuralNet::getBiasValue() const {
    return this->biasValue;
}

void NeuralNet::setBiasStatus(const bool useBias) {
    this->useBias = useBias;
}

bool NeuralNet::getBiasStatus() const {
    return this->useBias;
}

bool NeuralNet::save(const string& filename) {
    cout << "Exporting neural network " << this->name << " to " << filename << " ..." << endl;

    Json::Value jsonNN;
    jsonNN["useBias"] = this->useBias;
    jsonNN["biasValue"] = this->biasValue;
    jsonNN["layers"] = Json::Value(Json::arrayValue);

    for (size_t layerIndex = 0; layerIndex < this->numHiddenLayers + 2; layerIndex++) {
        Layer& layer = this->layers[layerIndex];
        Json::Value jsonLayer;

        if (layerIndex == 0)
            jsonLayer["type"] = "input";
        else if (layerIndex == this->numHiddenLayers + 1)
            jsonLayer["type"] = "output";
        else
            jsonLayer["type"] = "hidden";

        jsonLayer["hasBias"] = layer.hasBias;
        jsonLayer["neurons"] = Json::Value(Json::arrayValue);

        for (size_t neuron = 0; neuron < layer.numNeurons; neuron++) {
            Json::Value jsonNeuron;
            jsonNeuron["weights"] = Json::Value(Json::arrayValue);

            for (size_t weight = 0; weight < layer.neurons[neuron].numInputs; weight++)
                jsonNeuron["weights"].append(layer.neurons[neuron].weights[weight]);

            jsonLayer["neurons"].append(jsonNeuron);
        }

        jsonNN["layers"].append(jsonLayer);
    }

    ofstream out(filename.c_str());
    if (!out.is_open())
        return false;

    Json::FastWriter jsonWriter;
    out << jsonWriter.write(jsonNN);
    out.close();

    return true;
}

bool NeuralNet::load(const string& filename) {
    cout << "Loading " << filename << " ..." << endl;
    cerr << "Not implemented yet" << endl;
    return true;
}
