
#include <iostream>
#include <iomanip>
#include <random>
#include <thread>
#include <atomic>

#include "ComunicationStruct.hpp"
#include "TrainingLib.hpp"

void printCPPToPythonLive(int ix, const CPPToPython& data)
{
    // ANSI color codes
    constexpr const char* RESET  = "\033[0m";
    constexpr const char* TITLE  = "\033[1;36m";
    constexpr const char* LABEL  = "\033[36m";
    constexpr const char* VALUE  = "\033[33m";
    constexpr const char* HEADER = "\033[1;35m";

    constexpr int labelWidth = 28;
    constexpr int valueWidth = 10;

    static int lastLineCount = 0;

    // Move cursor up and clear previous output
    for (int i = 0; i < lastLineCount; ++i) {
        std::cout << "\033[1A\033[2K"; // cursor up + clear line
    }

    int lineCount = 0;

    auto printLine = [&](const std::string& s) {
        std::cout << s << "\n";
        ++lineCount;
    };

    std::cout << std::fixed << std::setprecision(3);

    printLine(std::string(TITLE) +
              "=== CPPToPython (ix = " + std::to_string(ix) + ") ===" +
              RESET);

	printLine("");

	printLine(std::string(LABEL) +
			  "Finished iterations: " +
			  VALUE + std::to_string(data.finishedIterations) +
			  RESET);

    printLine(std::string(LABEL) +
              "Finished networks this iter: " +
              VALUE + std::to_string(data.finishedNetworksThisIter) +
              RESET);

    printLine("");

    printLine(std::string(HEADER) + "Timing:" + RESET);

    auto timeLine = [&](const char* label, double value) {
        std::ostringstream oss;
        oss << "  "
            << LABEL << std::left << std::setw(labelWidth)
            << label << RESET
            << ": "
            << VALUE << std::right << std::setw(valueWidth)
            << value << " ms" << RESET;
        printLine(oss.str());
    };

	timeLine("timePerMutation", data.timePerMutate);
    timeLine("timePerNetwork", data.timePerNetwork);
    timeLine("timePerIter", data.timePerIter);
    timeLine("timeComputationNetwork", data.timeComputationNetwork);
    timeLine("timeComputationSimulation", data.timeComputetionSimulation);

    printLine("");

    std::ostringstream best;
    best << HEADER << "Best scores (" << data.bestScores.size() << "):" << RESET;
    printLine(best.str());

    {
        std::ostringstream oss;
        oss << "  ";
        for (double v : data.bestScores)
            oss << VALUE << v << " " << RESET;
        printLine(oss.str());
    }

    printLine("");

    std::ostringstream avg;
    avg << HEADER << "Average scores (" << data.avgScores.size() << "):" << RESET;
    printLine(avg.str());

    {
        std::ostringstream oss;
        oss << "  ";
        for (double v : data.avgScores)
            oss << VALUE << v << " " << RESET;
        printLine(oss.str());
    }

    lastLineCount = lineCount;
    std::cout << std::flush;
}

void printTimingOneLine(const CPPToPython& data)
{
	std::cout
		<< "Curr Iter: " << data.finishedNetworksThisIter
		<< "\n"
		<< std::flush;

	std::cout << data.finishedIterations << ": ";
	for (auto &v : data.bestScores) {
		std::cout << v << ", ";
	}
	std::cout << "\n";
}

std::atomic<bool> stopFlag{false};
void stoppingCondition() {
	std::cin.get();
	stopFlag = true;
}

int main() {

	PythonToCPP initSettings{
		"MNIST",
		{20, 20},
		50,
		1,
		20,
		0.1,
		0.2,
		0.3,
		0.1,
		0.09,
		0.9,
		1,
		1.0,
		{1, 0, 0, 0}
	};
	TrainingLib trainingLib{initSettings};

	std::jthread stopThread(stoppingCondition);

	int ix = 0;
	while (!stopFlag) {
		auto out = trainingLib.getInfo();
		printTimingOneLine(out);
		std::this_thread::sleep_for(std::chrono::milliseconds{500});
	}

	trainingLib.stop();

	// Mutation Step: 276 | Per Network: 11 | Complete Iteration: 11554 | comp(net): 14084 | comp(sim): 118 ms
	// Mutation Step: 268 | Per Network: 10 | Complete Iteration: 11072 | comp(net): 14094 | comp(sim): 122 ms
	// Mutation Step: 282 | Per Network:  2 | Complete Iteration:  2418 | comp(net):  2190 | comp(sim): 98 ms

	// USING MICROSECONDS NOW
	// Mutation Step: 281674 | Per Network: 2186 | Complete Iteration: 2.45567e+06 | comp(net): 2.21168e+06 | comp(sim):  98702 us
	// Mutation Step: 277687 | Per Network: 2231 | Complete Iteration:     2457620 | comp(net): 4.01443e+06 | comp(sim): 177598 us (support for threading)

	// USING THREADING
	// Mutation Step: 278486 | Per Network: 3792 | Complete Iteration: 531463 | comp(net): 7.51261e+07 | comp(sim): 2.86314e+06 us
	// Mutation Step: 284344 | Per Network: 3814 | Complete Iteration: 615487 | comp(net): 6.88778e+06 | comp(sim): 284046 us
	// Mutation Step:  59652 | Per Network: 2322 | Complete Iteration: 596091 | comp(net): 1.67127e+07 | comp(sim): 725161 us

	// MNIST
	// Mutation Step: 5.26135e+06 | Per Network: 876227 | Complete Iteration: 6.97054e+07 | comp(net): 5.15193e+08 | comp(sim): 4.76957e+06 us
}