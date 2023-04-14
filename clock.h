#ifndef CLOCK
#define CLOCK

#include <chrono>

class Clock {
	static std::chrono::steady_clock::time_point prevTime;
public:
	Clock();
	void updateClock();
	float getDeltaTime();
};

Clock::Clock() {
	prevTime = std::chrono::high_resolution_clock::now();
}

void Clock::updateClock() {
	prevTime = std::chrono::high_resolution_clock::now();
}

float Clock::getDeltaTime() {
	std::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(currentTime - prevTime).count()/1e6;
}

#endif
