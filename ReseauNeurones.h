// old version, new is NN2
#ifndef RESEAU_NEURONES_H
#define RESEAU_NEURONES_H

#include <vector>
#include <random>
#include <cmath>

#include <iostream>
/* non necessaire pour l'instant
#include <fstream>
#include <algorithm>
#include <iterator>
*/

/*
// includes for opencv use
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
*/

// Comme on travail en float on peut optimiser certains algos avec des parties x86
// Et préférablement en pact

//
class ReseauNeurones {
public:
	using real = float;
	using vec = std::vector<real>;
	using mat = std::vector<vec>;

	double err;

	ReseauNeurones( const vec& _input_layer,
			const std::vector<unsigned int> hidden_layer_shape,
			const unsigned int output_layer_size);

	// void input(vec input_layer); // inutile si input_layer est une ref constante
	void compute();
	void retropropagate(const vec& true_output_layer);
	const vec& output() const;

	static real sigmoid (real x);
	static real d_sigmoid (real x);
	
private:
	const vec& input_layer;
	vec input_layer_d;
	mat hidden_layers, hidden_layers_d;
	vec output_layer, output_layer_d;
	std::vector<mat> weight;
	mat input_layer_weight;

	real learning_rate = 1.0;


	// random
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen;
    std::uniform_real_distribution<real> dis;
};


#endif
