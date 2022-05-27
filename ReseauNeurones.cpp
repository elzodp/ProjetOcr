// old version, new is NN2
#include "ReseauNeurones.h"

using real = ReseauNeurones::real;
using vec = ReseauNeurones::vec;
using mat = ReseauNeurones::mat;

// XOR Table ?

/***
***	Constructeur du Reseau de neurones
***
***	Le vecteur &_input_layer ne doit pas être vide lors de l'appel
***		au constructeur car il est necessaire de connaître sa taille
***		pour l'initialisation de la 1ère couche de hidden_layer_shape
***
***	Les éléments de la matrice _weight sont initialisés aléatoirement
***
***	Membres à initialiser :
***			const vec& input_layer;
***			mat hidden_layers;
***			vec output_layer;
***			std::vector<mat> weight;
***			mat input_layer_weight;
***/
ReseauNeurones::ReseauNeurones(const vec& _input_layer,
							   const std::vector<unsigned int> hidden_layer_shape,
							   const unsigned int output_layer_size)
							   : input_layer(_input_layer)
							   , hidden_layers(hidden_layer_shape.size())
							   , hidden_layers_d(hidden_layer_shape.size())
							   , output_layer(output_layer_size)
							   , output_layer_d(output_layer_size)
							   , weight(hidden_layer_shape.size())
							   , input_layer_weight(_input_layer.size())
							   , gen(rd())
							   , dis(-1.0, 1.0)
							   , err(2e10) // mieux -> __DBL_MAX__
{
	for(auto &w_i : input_layer_weight)
		w_i = std::vector<real>(hidden_layer_shape[0]);

	for(int i = 0; ; i++){
		hidden_layers[i] = std::vector<real>(hidden_layer_shape[i]);
		hidden_layers_d[i] = std::vector<real>(hidden_layer_shape[i]);
		weight[i] = std::vector<vec>(hidden_layer_shape[i]);

		if(i >= hidden_layer_shape.size()-1) break;
		for( auto &x : weight[i])
			x = std::vector<real>(hidden_layer_shape[i+1]);
	}
	
	for( auto &x : weight[hidden_layer_shape.size()-1])
		x = std::vector<real>(output_layer_size);

	for( auto &vec_w : input_layer_weight)
		for( auto &w : vec_w)
			w = dis(gen); // on genere un real entre -1 et 1 pour l'input layer

	for( auto &mat_w : weight)
		for( auto &vec_w : mat_w)
			for( auto &w : vec_w)
				w = dis(gen); // on genere un real entre -1 et 1 pour l'input layer

}

/***
***	Propage le signal de l'input jusqu'à la sortie
***	Calcule le dot product entre le vecteur de la couche actuelle
***		et la matrice de ses poids associés puis applique à chaque
***		élement du vecteur resultant, la fonction sigmoid.
***/
void ReseauNeurones::compute()
{
	for(int i = 0; i < input_layer.size(); i++){
		real s = 0;
		for(int j = 0; j < weight[0].size(); j++){
			s += input_layer[i] * input_layer_weight[i][j];
		}
		hidden_layers[0][i] = sigmoid(s);
	}

	for(int k = 0; k < hidden_layers.size() - 1; k++)
		for(int i = 0; i < hidden_layers[k].size(); i++){
			real s = 0;
			for(int j = 0; j < weight[k].size(); j++){
				s += hidden_layers[k][i] * weight[k][i][j];
			}
			hidden_layers[k+1][i] = sigmoid(s);
		}
	
	const unsigned long long N = hidden_layers.size();
	for(int i = 0; i < hidden_layers[N-1].size(); i++){
		real s = 0;
		for(int j = 0; j < weight[N-1].size(); j++){
			s += hidden_layers[N-1][i] * weight[N-1][i][j];
		}
		output_layer[i] = sigmoid(s);
	}
}

/***
***	Modifie les poids de façon à minimiser l'erreur en utilisant
***	l'algorithme de la descente du gradient
***/
void ReseauNeurones::retropropagate(const vec& true_output_layer)
{
	const unsigned long long N = hidden_layers.size();
	err = 0;
	for(int i = 0; i < output_layer.size(); i++){
		// Mean absolute error
		err += std::abs(output_layer[i] - true_output_layer[i]);
		// output layer delta
		output_layer_d[i] = (output_layer[i] - true_output_layer[i]) * d_sigmoid(output_layer[i]);
	}

	for(int i = 0; i < hidden_layers[N-1].size(); i++)
		for(int j = 0; j < weight[N-1].size(); j++)
			hidden_layers_d[N-1][j] = output_layer_d[i] * weight[N-1][j][i] * d_sigmoid(hidden_layers[N-1][j]);

	for(int k = hidden_layers.size() - 2; k >= 0 ; k--){
		for(int i = 0; i < hidden_layers[k].size(); i++)
			for(int j = 0; j < weight[k].size(); j++)
				hidden_layers_d[k][j] = hidden_layers_d[k+1][i] * weight[k][j][i] * d_sigmoid(hidden_layers[k][j]);
	}
	/* // utile ????
	for(int i = 0; i < input_layer_d.size(); i++)
		for(int j = 0; j < input_layer_weight[i].size(); j++)
			input_layer_d[j] = hidden_layers_d[0][i] * input_layer_weight[j][i] * d_sigmoid(input_layer[j]);
	*/
	for(int i = 0; i < input_layer.size(); i++)
		for(int j = 0; j < hidden_layers_d[0].size(); j++)
			input_layer_weight[i][j] -= learning_rate * input_layer[i] * hidden_layers_d[0][j];

	for(int k = 0; k < hidden_layers.size() - 1; k++)
		for(int i = 0; i < hidden_layers[k].size(); i++)
			for(int j = 0; j < hidden_layers_d[k+1].size(); j++)
				weight[k][i][j] -= learning_rate * hidden_layers[k][i] * hidden_layers_d[k+1][j];

	for(int i = 0; i < hidden_layers[N-1].size(); i++)
		for(int j = 0; j < output_layer_d.size(); j++)
			weight[N-1][i][j] -= learning_rate * hidden_layers[N-1][i] * output_layer_d[j];
}

/***
***	Retourne une reference constante sur output_layer
***/
const vec &ReseauNeurones::output() const
{
	return output_layer;
}

/***
***	Retourne la valeur de la sigmoide de x :
***					1 / ( 1 + exp(-x) )
***/
real ReseauNeurones::sigmoid(real x)
{
	// Version raw
	return 1 / (1 + exp(-x));
}

/***
*** Retourne la valeur de la dérivé de sigmoid de x en fonction
***		de sigmoid de x
***/
real ReseauNeurones::d_sigmoid(real x)
{
	return x * (1 - x);
}

//image processing functions

/*
using namespace cv;

std::vector<int> imgtoinput (string input_path)
{
	//loading the image from path and getting its size
	Mat imbrg = imread(input_path, IMREAD_COLOR);
	int L = imbrg.rows;
	int C = imbrg.cols;
	//declaration of mat and convertion to grayscale
	Mat imgray;
	cv::cvtColor(imbrg, imgray, cv::COLOR_BGR2GRAY,0);
	// declaration of vector and setting its capacity
	std::vector<int> vct;
	vct.reserve(L*C);
	//starting loops
	for(int i=0; i<L; i++)
	{
		for(int j=0; j<C; j++)
		{
			vct.push_back((int)imgray.at<uchar>(i,j));
		}
	}
	return vct ;
}
*/
