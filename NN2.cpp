#include "NN2.h"

using real = NN2::real;
using vec = NN2::vec;
using mat = NN2::mat;

#define MSG_ERR(M) std::cout << M << std::endl;
//int c = 0;

/***
 * 	Constructeur du Reseau de neurones
 * 
 * 	- shape est une liste de la taille de chaque couche
 *    ex: {3,5,4,1}		input layer 	: 3 elements
 * 						hidden layer 1 	: 5 elements
 * 						hidden layer 2 	: 4 elements
 * 						output layer 	: 1 elements 
 * 
 *  - nbr_datas est le nombre d'éléments sur lesquels 
 *  s'entraine le reseau de neuronnes
***/
NN2::NN2(const std::vector<unsigned int> shape, const unsigned int nbr_datas)
							: shape(shape)
							, nbr_datas(nbr_datas)
							, layers(nbr_datas)
							, layers_d(nbr_datas)
							, weight(shape.size()-1)
							, gen(rd())
							, dis(-1.0, 1.0)
							, err(2e10)
							, learning_rate(1.0)
{
    assert(shape.size() >= 3);

	for(int l = 0; l < nbr_datas; l++){
		layers[l] = mat(shape.size());
		layers_d[l] = mat(shape.size());

		for(int i = 0; i < shape.size(); i++){
			layers[l][i] = vec(shape[i]);
			layers_d[l][i] = vec(shape[i]);
		}
	}
	
	for(int i = 0; i < shape.size() - 1; i++){
		weight[i] = mat(shape[i]);
		for( auto &w_ij : weight[i])
			w_ij = vec(shape[i+1]);
	}

    // initialisation des poids
	for( auto &mat_w : weight)
		for( auto &vec_w : mat_w)
			for( auto &w : vec_w)
				w = dis(gen);
}

/***
 * 	Propage le signal de l'input jusqu'à la sortie
 * 	Calcule le dot product entre le vecteur de la couche actuelle
 * 		et la matrice de ses poids associés puis applique à chaque
 * 		élement du vecteur resultant, la fonction sigmoid.
***/
void NN2::compute(const mat& input)
{
	for(int l = 0; l < nbr_datas; l++){
		layers[l][0] = input[l];

		for(int k = 0; k < shape.size() - 1; k++){
			for(int j = 0; j < shape[k+1]; j++){
				real s = 0.0;
				for(int i = 0; i < shape[k]; i++){
					s += layers[l][k][i] * weight[k][i][j];
				}
				layers[l][k+1][j] = sigmoid(s);
			}
		}
	}
}
/*** 
 *  compute un seul vecteur input, dans layers[0]
***/
void NN2::compute(const vec& input)
{
	layers[0][0] = input;

	for(int k = 0; k < shape.size() - 1; k++){
		for(int j = 0; j < shape[k+1]; j++){
			real s = 0.0;
			for(int i = 0; i < shape[k]; i++){
				s += layers[0][k][i] * weight[k][i][j];
			}
			layers[0][k+1][j] = sigmoid(s);
		}
	}
}

/***
 * 	Modifie les poids de façon à minimiser l'erreur en utilisant
 * 	l'algorithme de la descente du gradient
***/
void NN2::retropropagate(const mat& exact_result)
{
	const unsigned int N = shape.size() - 1; // last layer index
	err = 0;

    // calcul erreur + output layer d
	for(int l = 0; l < nbr_datas; l++){
		for(int i = 0; i < shape[N]; i++){
			const real diff = layers[l][N][i] - exact_result[l][i];
			err += std::abs(diff); // Tneter avec l'erreure quadratique
			layers_d[l][N][i] = diff * d_sigmoid(layers[l][N][i]);
		}
	}

	err /= nbr_datas; // normalisation de l'erreur

    // descente
	for(int l = 0; l < nbr_datas; l++){
		for(int k = N - 1; k > 0 ; k--){
			for(int i = 0; i < shape[k]; i++){
				double s = 0.0;
				for(int j = 0; j < shape[k+1]; j++)
					s += layers_d[l][k+1][j] * weight[k][i][j];
				layers_d[l][k][i] = s * d_sigmoid(layers[l][k][i]); // Tenter avec une autre fonction que sigmoid
			}
		}
	}

    // ajustement des poids
	for(int k = 0; k < N; k++)
		for(int i = 0; i < shape[k]; i++)
			for(int j = 0; j < shape[k+1]; j++){
				real s = 0.0;
				for(int l = 0; l < nbr_datas; l++)
					s += layers[l][k][i] * layers_d[l][k+1][j];
				weight[k][i][j] -= learning_rate * s;
			} // TODO: la couche k=0 de layers_d n'est jamais utiliser, à suppr
}

/***
 * Retourne une reference constante sur la matrice
 * des poids
***/
const std::vector<mat>& NN2::get_weight() const{
	return weight;
}

/***
 * Retourne une reference constante sur la matrice
 * des poids
***/
void NN2::set_weight(const std::vector<mat>& w){
	weight = w;
}


/***
 * 	Retourne une reference constante sur layers[0][N]
 *  cad le resultat du premier element passé en input
 *  à la fonction compute
***/
const vec &NN2::output() const
{
	return layers[0][shape.size() - 1];
}

/***
 * 	Retourne une reference constante sur l'erreur
***/
const double &NN2::get_err() const
{
	return err;
}

/***
 * 	Retourne la valeur de la sigmoide de x :
 * 					1 / ( 1 + exp(-x) )
***/
real NN2::sigmoid(real x)
{
	// Version raw
	return 1 / (1 + exp(-x));
}

/***
 *  Retourne la valeur de la dérivé de sigmoid de x en fonction
 * 		de sigmoid de x
***/
real NN2::d_sigmoid(real x)
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
// Lecture en colonne ou ligne des pixels, qui seront rentrés en input
