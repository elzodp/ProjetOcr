#include "test_xor.h"
#include <ctime>
#include <chrono>

void Test_XOR::run_test(){
        NN2 nn({3,5,1}, 8); // Pourquoi un certains nombre de couches? Combien de neurones par coucjes
        auto start = std::chrono::steady_clock::now();
        for(int i = 0; i < 10e5; i++){
            // int n = static_cast<int>(I.size() * rand()/RAND_MAX);
            nn.compute(I);
            nn.retropropagate(O);

            if(i%static_cast<int>(10e4) == 0 || i < 10)
                std::cout << "err = " << nn.get_err() << std::endl;
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
      int a, b, c;
      std::cout << "Test value (for exemple : 1 1 0)" << std::endl;
      std::cout << "--> ";
      std::cin >> a >> b >> c;
      real ra = a, rb = b, rc = c;
      nn.compute({ra,rb,rc});
      // auto r =
      std::cout << "result : ";
      std::cout << nn.output()[0] << std::endl;
      std::cout << "exact result : " << a << "^" << b << "^" << c << " = " << ((a^b)^c)
                << std::endl << std::endl;

}

Test_XOR::Test_XOR() : I(8), O(8){

    for(auto &i : I)
        i = vec(3);
    for(auto &o : O)
        o = vec(1);
    //XOR tables
    I[0][0] = 0;
    I[0][1] = 0;
    I[0][2] = 0;
    O[0][0] = 0;

    I[1][0] = 1;
    I[1][1] = 1;
    I[1][2] = 1;
    O[1][0] = 1;

    I[2][0] = 0;
    I[2][1] = 0;
    I[2][2] = 1;
    O[2][0] = 1;

    I[3][0] = 1;
    I[3][1] = 1;
    I[3][2] = 0;
    O[3][0] = 0;

    I[4][0] = 0;
    I[4][1] = 1;
    I[4][2] = 0;
    O[4][0] = 1;

    I[5][0] = 1;
    I[5][1] = 0;
    I[5][2] = 1;
    O[5][0] = 0;

    I[6][0] = 1;
    I[6][1] = 0;
    I[6][2] = 0;
    O[6][0] = 1;

    I[7][0] = 0;
    I[7][1] = 1;
    I[7][2] = 1;
    O[7][0] = 0;
}

// XOR trop simple pour plusieurs sous couches
