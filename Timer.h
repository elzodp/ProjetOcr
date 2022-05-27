#pragma once
#include <chrono>
#include <iostream>

class Timer {
public:

  void start();
  void end();

  std::chrono::microseconds  getDuration();

private:
  std::chrono::high_resolution_clock::time_point begin_t, end_t;
  
};