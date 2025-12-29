
#include <iostream>
#include <iomanip>
#include <random>
#include <thread>
#include <atomic>

#include "ComunicationStruct.hpp"
#include "TrainingLib.hpp"

void printCPPToPythonOld(const CPPToPython& data) {
	// This function is created by ChatGPT
	std::cout << "=== CPPToPython ===\n";

	std::cout << "Finished iterations: "
			  << data.finishedIterations << "\n";

	std::cout << "Finished networks this iter: "
			  << data.finishedNetworksThisIter << "\n";

	std::cout << "Timing:\n";
	std::cout << "  timePerNetwork            : " << data.timePerNetwork << "\n";
	std::cout << "  timePerIter               : " << data.timePerIter << "\n";
	std::cout << "  timeComputationNetwork    : " << data.timeComputationNetwork << "\n";
	std::cout << "  timeComputationSimulation : " << data.timeComputetionSimulation << "\n";

	std::cout << "Best scores (" << data.bestScores.size() << "): ";
	for (double v : data.bestScores)
		std::cout << v << " ";
	std::cout << "\n";

	std::cout << "Average scores (" << data.avgScores.size() << "): ";
	for (double v : data.avgScores)
		std::cout << v << " ";
	std::cout << "\n";

	std::cout << "===============================\n";
}

void printCPPToPython(int ix, const CPPToPython& data) {
	// This function is written by ChatGPT
	// ANSI color codes
	constexpr const char* RESET  = "\033[0m";
	constexpr const char* TITLE  = "\033[1;36m"; // bold cyan
	constexpr const char* LABEL  = "\033[36m";   // cyan
	constexpr const char* VALUE  = "\033[33m";   // yellow
	constexpr const char* HEADER = "\033[1;35m"; // bold magenta

	constexpr int labelWidth = 28;
	constexpr int valueWidth = 10;

	std::cout << TITLE
			  << "=== CPPToPython (ix = " << ix << ") ==="
			  << RESET << "\n\n";

	std::cout << LABEL << "Finished networks this iter: "
			  << VALUE << data.finishedNetworksThisIter
			  << RESET << "\n\n";

	std::cout << HEADER << "Timing:\n" << RESET;

	std::cout << std::fixed << std::setprecision(3);

	auto printTime = [&](const char* label, double value) {
		std::cout << "  "
				  << LABEL << std::left << std::setw(labelWidth)
				  << label << RESET
				  << ": "
				  << VALUE << std::right << std::setw(valueWidth)
				  << value << " ms" << RESET << "\n";
	};

	printTime("timePerNetwork", data.timePerNetwork);
	printTime("timePerIter", data.timePerIter);
	printTime("timeComputationNetwork", data.timeComputationNetwork);
	printTime("timeComputationSimulation", data.timeComputetionSimulation);

	std::cout << "\n" << HEADER
			  << "Best scores (" << data.bestScores.size() << "):"
			  << RESET << "\n  ";

	for (double v : data.bestScores)
		std::cout << VALUE << v << " " << RESET;

	std::cout << "\n\n" << HEADER
			  << "Average scores (" << data.avgScores.size() << "):"
			  << RESET << "\n  ";

	for (double v : data.avgScores)
		std::cout << VALUE << v << " " << RESET;

	std::cout << "\n\n";
}

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

std::atomic<bool> stopFlag{false};
void stoppingCondition() {
	std::cin.get();
	stopFlag = true;
}

int main() {

	PythonToCPP initSettings{
		"Pendulum",
		{20, 50, 20},
		1000,
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
		printCPPToPythonLive(ix++, out);
		std::this_thread::sleep_for(std::chrono::milliseconds{5000});
	}

	trainingLib.stop();
}