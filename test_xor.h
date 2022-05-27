#include "NN2.h"
#include <iostream>

class Test_XOR{
    using real = NN2::real;
    using vec = NN2::vec;
    using mat = NN2::mat;

    public:
    mat I;
    mat O;
    Test_XOR();
    void run_test();
    
};
