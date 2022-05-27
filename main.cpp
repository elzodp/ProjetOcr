#include <iostream>
#include <ctime>
#include <chrono>
// #include <vector>
// #include <fstream>
// #include "tools.h"
#include "Timer.h"

#include "test_xor.h"
#include "NN2.h"
//
//

// On peut utiliser çà pour lire les images et surement ranger les données en black and white dans un file output.dat que le réseaux peut lire ou bien on peut le faire dans tools.h avec un void image() et il faudra resize
int main()
{
    // ifstream image;
    // image.open("C:\\Home\\ProjetOCR\\src\\test_numlet1.png");
    // image.read();

  NN2 nn({3,5,1}, 8); // reseau : 3 - 5 - 1

  Test_XOR test;

	//Lancé le réseau plusieurs fois avec une boucle for?
	Timer t;
	// Itération de 0 à n
	double temps =0.0;

	FILE* fp;
	const char* filename = "exec.dat";
	fp = fopen(filename, "w");
	if (!fp)
		{std::cout<<"No data file opened!"<<std::endl;}

  int work = 0; // 0 for training and 1 for working
  if (work == 0)
  {
    for (int iteration=0; iteration<10; iteration ++)
    {
      t.start();
      test.run_test();
      // nn.compute(input);
      // nn.compute(output);
      t.end();
      auto elapsed_time = t.getDuration();
      temps=temps+elapsed_time.count();
      fprintf(fp,"%d %lf", iteration, temps); // gnuplot
    }
  }
  else
  {
    t.start();
    test.run_test();
    // nn.compute(input);
    // nn.compute(output);
    t.end();
    auto elapsed_time = t.getDuration();
    temps=elapsed_time.count();
    printf("\n Le temps pour faire cette OCR est : %f \n",temps);
  }
	fclose(fp);
    return 0;
}
