#include <string>
#include "base/cuda_timer.h"
#include <cuda_runtime_api.h>

CudaTimer::CudaTimer() : elapsedTime_(0.0f), isRunning_(false) {
    cudaError_t err = cudaEventCreate(&start_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create start event: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    err = cudaEventCreate(&stop_);
    if (err != cudaSuccess) {
        cudaEventDestroy(start_);
        throw std::runtime_error("Failed to create stop event: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void CudaTimer::start(){
    if (isRunning_) {
        std::cerr << "Warning: Timer is already running!" << std::endl;
        return;
    }
    cudaEventRecord(start_);
    isRunning_ = true;
}

void CudaTimer::stop(){
    if (!isRunning_) {
        std::cerr << "Warning: Timer is not running!" << std::endl;
        return;
    }
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    isRunning_ = false;
    
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_, stop_);
    elapsedTime_ += milliseconds;
}

void CudaTimer::reset(){
    elapsedTime_ = 0.0f;
    isRunning_ = false;
}

float CudaTimer::elapsed() const{
    return elapsedTime_;
}

void CudaTimer::print(const std::string& message) const{
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    std::cout << elapsedTime_ << " ms" << std::endl;
}