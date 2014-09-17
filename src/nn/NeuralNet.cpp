/**
* NeuralNet
*
* This class can be used to create a neural network with a given size and provides the backpropagation algorithm and
* funtions to save and load a neural network from an XML file.
*
* @author Shivan Taher
* @date 22.03.2009
*/

#include "NeuralNetModule.h"

NeuralNet::NeuralNet() {
	this->learningRate = 1;
	this->biasValue = 1;
	this->useBias = true;
}

NeuralNet::~NeuralNet() {
	srand(time(0));
}

void NeuralNet::setNumInputs(int n) {
	this->numInputs = n;
}

int NeuralNet::getNumInputs() {
	return this->numInputs;
}

void NeuralNet::setNumOutputs(int n) {
	this->numOutputs = n;
}

int NeuralNet::getNumOutputs() {
	return this->numOutputs;
}

void NeuralNet::setNumHiddenLayers(int n) {
	this->numHiddenLayers = n;
}

int NeuralNet::getNumHiddenLayers() {
	return this->numHiddenLayers;
}

void NeuralNet::setNumNeuronsPerHL(int n) {
	this->numNeuronsPerHL = n;
}

int NeuralNet::getNumNeuronsPerHL() {
	return this->numNeuronsPerHL;
}

void NeuralNet::createNet() {
	// Create the input layer
	NeuralLayer inputLayer(this->numInputs, 0, false); // Ein Input-Neuron hat selber keine Inputs
	this->layers.push_back(inputLayer);

	// Create the hidden layers
	for (int n = 0; n < this->numHiddenLayers; ++n) {
		if (n == 0) {
			NeuralLayer hiddenLayer(this->numNeuronsPerHL, this->numInputs, this->useBias);
			this->layers.push_back(hiddenLayer);
		} else {
			NeuralLayer hiddenLayer(this->numNeuronsPerHL, this->numNeuronsPerHL, this->useBias);
			this->layers.push_back(hiddenLayer);
		}
	}

	// Create the output layer
	NeuralLayer outputLayer(this->numOutputs, this->numNeuronsPerHL, this->useBias);
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

vector<double> NeuralNet::calculateOutputs(vector<double> inputs) {
	// Results
	this->outputs.clear();

	// Check the size of the inputs
	if (inputs.size() != this->numInputs)
		return outputs;

	// For each layer
	for (int i = 0; i < (int) this->layers.size(); ++i) {
		if (i > 0) // The input is the output of the last layer
			inputs = this->outputs;

		this->outputs.clear();

		// Calculate the outputs = sigmoid(sum of (inputs * weights))

		for (int j = 0; j < this->layers[i].numNeurons; ++j) {
			double netinput = 0;

			int numInputs = this->layers[i].neurons[j].numInputs;

			// For each weight
			// Calculate the net input (sum of inputs * weights)

			// Ignore the input layer
			if (i > 0) {
				for (unsigned int k = 0; k < inputs.size(); ++k)
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

vector<double> NeuralNet::getOutputs() {
	return this->outputs;
}

double NeuralNet::backpropagation(vector<double> inputs, vector<double> expectedOutputs) {
	// Error values
	vector<double> delta_j, delta_i;
	double standardError = 0;

	// Calculate the activity of the network first
	this->calculateOutputs(inputs);

	//
	// Calculate and correct the errors of the output unit
	//

	for (int i = 0; i < this->numOutputs; i++) {
		// Err = y - aj = y - g(net_j)
		double err = expectedOutputs[i] - this->outputs[i];
		standardError += err;

		// netinput of the neuron i
		double net_i = this->layers[this->numHiddenLayers + 1].neurons[i].netInput;

		double di = err * this->sigmoidDerivation(net_i);
		delta_i.push_back(di);

		// Correct the weights between the output layer and the hidden layer

		int numInputsI = this->layers[this->numHiddenLayers + 1].neurons[i].numInputs;
		for (int j = 0; j < numInputsI; j++) {
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

	for (int L = this->numHiddenLayers; L > 0; L--) {
		for (int j = 0; j < this->numNeuronsPerHL; j++) {
			double err_j = 0;

			// Calculate the errors of the neuron j
			for (int i = 0; i < this->layers[L + 1].numNeurons; i++) {
				err_j += this->layers[L + 1].neurons[i].weights[j] * delta_i[i];
			}

			double net_j = this->layers[L].neurons[j].netInput;
			double dj = this->sigmoidDerivation(net_j) * err_j;
			delta_j.push_back(dj);

			// Correct the weights between the hidden layer and the predecessor layer

			int numInputsJ = this->layers[L].neurons[j].numInputs;
			for (int k = 0; k < numInputsJ; k++) {
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

double NeuralNet::getLearningRate() {
	return this->learningRate;
}

void NeuralNet::setMomentum(double value) {
	this->momentum = value;
}

double NeuralNet::getMomentum() {
	return this->momentum;
}

void NeuralNet::setBiasValue(double bias) {
	this->biasValue = bias;
}

double NeuralNet::getBiasValue() {
	return this->biasValue;
}

void NeuralNet::setBiasStatus(bool useBias) {
	this->useBias = useBias;
}

bool NeuralNet::getBiasStatus() {
	return this->useBias;
}

bool NeuralNet::saveFile(string filename) {
	FILE *file = fopen(filename.c_str(), "w+");
	if (!file)
		return false;
	string xml_content = "<?xml version=\"1.0\" ?>\n";
	xml_content += "<NeuralNet useBias=\"";
	xml_content += (this->useBias == true ? "1" : "0");
	xml_content += "\" biasValue=\"";
	xml_content += doubleToString(this->biasValue);
	xml_content += "\">\n";

	for (int layer = 0; layer < this->numHiddenLayers + 2; layer++) {
		if (layer == 0)
			xml_content += "\t<Layer type=\"input\" ";
		else if (layer == this->numHiddenLayers + 1)
			xml_content += "\t<Layer type=\"output\" ";
		else
			xml_content += "\t<Layer type=\"hidden\" ";

		if (this->layers[layer].hasBias)
			xml_content += "hasBias=\"1\">\n";
		else
			xml_content += "hasBias=\"0\">\n";

		for (int neuron = 0; neuron < this->layers[layer].numNeurons; neuron++) {
			xml_content += "\t\t<Neuron>\n";
			for (int weight = 0; weight < this->layers[layer].neurons[neuron].numInputs; weight++) {
				xml_content += "\t\t\t<weight value=\"";
				xml_content += doubleToString(this->layers[layer].neurons[neuron].weights[weight]);
				xml_content += "\" />\n";
			}
			xml_content += "\t\t</Neuron>\n";
		}

		xml_content += "\t</Layer>\n";
	}

	xml_content += "</NeuralNet>";

	fwrite(xml_content.c_str(), xml_content.length(), 1, file);
	fclose(file);
	return true;
}

bool NeuralNet::loadFile(string filename) {
	TiXmlDocument doc(filename.c_str());
	if (!doc.LoadFile())
		return false;

	this->layers.clear();
	this->numHiddenLayers = 0;
	this->numInputs = 0;
	this->numNeuronsPerHL = 0;
	this->numOutputs = 0;

	TiXmlHandle hDoc(&doc);
	TiXmlElement *elemLayer, *elemNeuron, *elemWeight, *elemNN;
	TiXmlHandle hRoot(0), hLayer(0), hNeuron(0), hWeight(0);

	elemLayer = hDoc.FirstChildElement().Element();

	// should always have a valid root but handle gracefully if it doesn't

	if (!elemLayer) return false;
	string m_name = elemLayer->Value();

	// save this for later
	hRoot = TiXmlHandle(elemLayer);
	elemNN = hRoot.Element();

	int bias = 1;
	double biasV = 1;
	if (elemNN->Attribute("useBias", (int *) &bias) != 0) {
		this->useBias = (bool) bias;
	}

	if (elemNN->Attribute("biasValue", (double *) &biasV) != 0) {
		this->biasValue = biasV;
	}

	// For each layer
	for (int i = 0; ; i++) {
		hLayer = hRoot.Child(i);
		elemLayer = hLayer.Element();
		if (elemLayer == 0)
			break;


		int layerBias = 0;
		const char *layerType = elemLayer->Attribute("type");
		elemLayer->Attribute("hasBias", (int *) &layerBias);

		NeuralLayer layer;

		if (layerBias == 1)
			layer.hasBias = true;
		else
			layer.hasBias = false;

		// For each neuron
		for (int j = 0; ; j++) {
			hNeuron = hLayer.Child("Neuron", j);
			elemNeuron = hNeuron.Element();
			if (elemNeuron == 0)
				break;

			Neuron neuron;

			for (int w = 0; ; w++) {
				hWeight = hNeuron.Child("weight", w);
				elemWeight = hWeight.Element();
				if (elemWeight == 0)
					break;

				double weight = 0;
				elemWeight->Attribute("value", (double *) &weight);

				neuron.numInputs++;
				neuron.weights.push_back(weight);
				neuron.deltaWeights.push_back(0);
			}

			layer.neurons.push_back(neuron);
			layer.numNeurons++;
		}

		this->layers.push_back(layer);


		if (i == 0) {
			this->numInputs = layer.numNeurons;
			//cout<<"Input layer loaded with "<<this->numInputs<<" Inputs..\n\n";
		} else if (strcmp(layerType, "hidden") == 0) {
			this->numHiddenLayers++;
			this->numNeuronsPerHL = layer.numNeurons;
			//cout<<"Hidden with "<<this->numNeuronsPerHL<<" Neurons..\n\n";
		} else if (strcmp(layerType, "output") == 0) {
			this->numOutputs = layer.numNeurons;
			//cout<<"Output with "<<this->numOutputs<<" Neurons..\n\n";
		}
		//cout<<"\n\n";
	}

	return true;
}