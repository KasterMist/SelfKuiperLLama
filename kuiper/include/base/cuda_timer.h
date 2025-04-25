#include <iostream>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

class CudaTimer{
public:
    CudaTimer();

    ~CudaTimer();

    void start();

    void stop();

    void reset();

    float elapsed() const;

    void print(const std::string& message = "") const;

    // 禁用拷贝和赋值
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    float elapsedTime_;  // 累计时间(毫秒)
    bool isRunning_;     // 计时器是否正在运行
};