#include "Timer.h"
#include <chrono>
#include <iostream>

void Timer::start() { begin_t = std::chrono::high_resolution_clock::now(); }

void Timer::end() { end_t = std::chrono::high_resolution_clock::now(); }

std::chrono::microseconds Timer::getDuration() {
  auto elapsed = end_t - begin_t;
  return std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
}