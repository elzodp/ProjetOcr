#ifndef NN2_H
#define NN2_H

#include <vector>
#include <random>
#include <cmath>
#include <cassert>

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
class NN2 {
public:
	using real = double;
	using vec = std::vector<real>;
	using mat = std::vector<vec>;


	NN2(const std::vector<unsigned int> shape, const unsigned int nbr_datas);

	void compute(const mat& input);
	void compute(const vec& input);
	void retropropagate(const mat& exact_result);

	const vec& output() const;
	const double& get_err() const;
	const std::vector<mat>& get_weight() const;
	void set_weight(const std::vector<mat>& w);

	static real sigmoid (real x);
	static real d_sigmoid (real x);
	
private:
	std::vector<mat> layers, layers_d;
	std::vector<mat> weight;

	const std::vector<unsigned int> shape;
	const unsigned int nbr_datas;
	double learning_rate;
	double err;

	// random
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen;
    std::uniform_real_distribution<real> dis;
};


#endif
