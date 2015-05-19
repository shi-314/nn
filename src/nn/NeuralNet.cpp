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

NeuralNet::NeuralNet(const string& name)
    : numHiddenLayers(0),
      momentum(0.9),
      learningRate(1),
      biasValue(1),
      useBias(true),
      name(name)
{
}

NeuralNet::~NeuralNet() {
    for(Layer* layer : this->layers)
        delete layer;
}

void NeuralNet::add(Layer::Type layerType, size_t numNeurons) {
    if (layerType == Layer::INPUT) {
        // Create the input layer
        this->layers.push_back(new Layer(numNeurons, 0, false));
    } else if (layerType == Layer::HIDDEN) {
        // Create the hidden layers
        this->numHiddenLayers++;
        Layer* lastLayer = this->layers[this->layers.size() - 1];
        this->layers.push_back(new Layer(numNeurons, lastLayer->numNeurons, this->useBias));
    } else if (layerType == Layer::OUTPUT) {
        // Create the output layer
        Layer* lastLayer = this->layers[this->layers.size() - 1];
        this->layers.push_back(new Layer(numNeurons, lastLayer->numNeurons, this->useBias));
    }
}

double NeuralNet::sigmoid(double x) {
    //double response = 1;
    //return 1/(1+exp(-x/response));
    // return 1 / (1 + exp(-x));
    return max(0.0, x);
}

double NeuralNet::sigmoidDerivation(double x) {
    // return log(1 + exp(x));

    // f'(x) = e^x / (e^x+1) = 1 / (1 + e^{-x})
    // Derivation of softplus function
    return 1.0 / (1.0 + exp(-x));

    // double gx = this->sigmoid(x);
    // return gx * (1 - gx);
}

const vector<double>& NeuralNet::calculateOutputs(vector<double> inputs) {
    // Results
    this->outputs.clear();

    // Check the size of the inputs
    if (inputs.size() != this->layers[0]->numNeurons)
        return outputs;

    // For each layer
    for (size_t i = 0; i < this->layers.size(); ++i) {
        if (i > 0) // The input is the output of the last layer
            inputs = this->outputs;

        this->outputs.clear();
        
        Layer* li = this->layers[i];

        // Calculate the outputs = sigmoid(sum of (inputs * weights))

        for (size_t j = 0; j < li->numNeurons; ++j) {
            Neuron* nj = li->neurons[j];

            double netinput = 0;

            size_t numInputs = nj->numInputs;

            // For each weight
            // Calculate the net input (sum of inputs * weights)

            // Ignore the input layer
            if (i > 0) {
                for (size_t k = 0; k < inputs.size(); ++k)
                    netinput += nj->weights[k] * inputs[k];
            } else {
                netinput = inputs[j];
            }

            // Add the bias value if enabled
            if (this->useBias && li->hasBias) {
                netinput += nj->weights[numInputs - 1] * this->biasValue;
            }

            nj->netInput = netinput;
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

    Layer* outputLayer = this->layers[this->layers.size() - 1];

    for (size_t i = 0; i < outputLayer->numNeurons; i++) {
        Neuron* ni = outputLayer->neurons[i];

        // Err = y - aj = y - g(net_j)
        double err = expectedOutputs[i] - this->outputs[i];
        standardError += err;

        // netinput of the neuron i
        double net_i = ni->netInput;

        double di = err * this->sigmoidDerivation(net_i);
        delta_i.push_back(di);

        // Correct the weights between the output layer and the hidden layer

        size_t numInputsI = ni->numInputs;
        for (size_t j = 0; j < numInputsI; j++) {
            double net_j;

            // Does the layer have a bias and is j the bias neuron?
            if (outputLayer->hasBias && j == numInputsI - 1) {
                net_j = this->biasValue;
            } else {
                net_j = this->layers[this->numHiddenLayers]->neurons[j]->netInput;
            }

            double delta_w = this->learningRate * this->sigmoid(net_j) * di + this->momentum * ni->deltaWeights[j];

            ni->deltaWeights[j] = delta_w;
            ni->weights[j] += delta_w;
        }
    }

    standardError /= outputLayer->numNeurons;
    standardError = (standardError * standardError) / 2; // E = 1/2 Err^2

    //
    // Calculate and correct the errors of the hidden layers
    //

    for (size_t L = this->numHiddenLayers; L > 0; L--) {
        Layer* hl = this->layers[L];
        Layer* prevHl = this->layers[L - 1];
        Layer* nextHl = this->layers[L + 1];

        for (size_t j = 0; j < hl->numNeurons; j++) {
            Neuron* nj = hl->neurons[j];
            double err_j = 0;

            // Calculate the errors of the neuron j
            for (size_t i = 0; i < nextHl->numNeurons; i++) {
                err_j += nextHl->neurons[i]->weights[j] * delta_i[i];
            }

            double net_j = nj->netInput;
            double dj = this->sigmoidDerivation(net_j) * err_j;
            delta_j.push_back(dj);

            // Correct the weights between the hidden layer and the predecessor layer

            size_t numInputsJ = nj->numInputs;
            for (size_t k = 0; k < numInputsJ; k++) {
                double net_k;

                // Does the layer have a bias and is j the bias neuron?
                if (hl->hasBias && k == numInputsJ - 1) {
                    net_k = this->biasValue;
                } else {
                    net_k = prevHl->neurons[k]->netInput;
                }
                double delta_w = this->learningRate * this->sigmoid(net_k) * dj + this->momentum * nj->deltaWeights[k];

                hl->neurons[j]->deltaWeights[k] = delta_w;
                hl->neurons[j]->weights[k] += delta_w;
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
        Layer* layer = this->layers[layerIndex];
        Json::Value jsonLayer;

        if (layerIndex == 0)
            jsonLayer["type"] = "input";
        else if (layerIndex == this->numHiddenLayers + 1)
            jsonLayer["type"] = "output";
        else
            jsonLayer["type"] = "hidden";

        jsonLayer["hasBias"] = layer->hasBias;
        jsonLayer["neurons"] = Json::Value(Json::arrayValue);

        for (size_t i = 0; i < layer->numNeurons; i++) {
            Json::Value jsonNeuron;
            jsonNeuron["weights"] = Json::Value(Json::arrayValue);

            for (size_t j = 0; j < layer->neurons[i]->numInputs; j++)
                jsonNeuron["weights"].append(layer->neurons[i]->weights[j]);

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
