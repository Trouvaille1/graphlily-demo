#ifndef DEMO_BASE_MODULE_H_
#define DEMO_BASE_MODULE_H_

#include "demo/global.h"


namespace demo {
namespace module {

//抽象类，不能实例化，只能被继承
class BaseModule {
protected:
    std::string kernel_name_;//kernel的名字
    std::string target_;//目标设备.可以是sw_emu, hw_emu, hw

    // OpenCL runtime
    cl::Device device_;
    cl::Context context_;
    cl::Kernel kernel_;
    cl::CommandQueue command_queue_;

public:
    BaseModule(std::string kernel_name) {
        this->kernel_name_ = kernel_name;
    }

    virtual ~BaseModule() {
        this->device_ = nullptr;
        this->context_ = nullptr;
        this->kernel_ = nullptr;
        this->command_queue_ = nullptr;
    }

    std::string get_kernel_name() {
        return this->kernel_name_;
    }

    void set_target(std::string target) {
        assert(target == "sw_emu" || target == "hw_emu" || target == "hw");
        this->target_ = target;
    }    

    void set_device(cl::Device device) {
        this->device_ = device;
    }

    void set_context(cl::Context context) {
        this->context_ = context;
    }

    void set_kernel(cl::Kernel kernel) {
        this->kernel_ = kernel;
    }

    void set_command_queue(cl::CommandQueue command_queue) {
        this->command_queue_ = command_queue;
    }

    //在设备上从一个buffer复制到另一个buffer，无需经过主机
    void copy_buffer_device_to_device(cl::Buffer src, cl::Buffer dst, size_t bytes) {
        this->command_queue_.enqueueCopyBuffer(src, dst, 0, 0, bytes);
        this->command_queue_.finish();
    }

    //设置不需要的参数。纯虚函数，子类必须重写
    virtual void set_unused_args() = 0;

    //设置模式。SpMV和SpMSpV合并到一个kernel中；我们需要选择其中一个，所以称为模式。同样，所有的赋值(apply)函数都合并到一个kernel中。
    virtual void set_mode() = 0;

    //加载xclbin文件并设置运行时
    void set_up_runtime(std::string xclbin_file_path);
};

//设置OpenCL runtime
void BaseModule::set_up_runtime(std::string xclbin_file_path) {
    cl_int err;
    
    if (this->target_ == "sw_emu" || this->target_ == "hw_emu") {
        setenv("XCL_EMULATION_MODE", this->target_.c_str(), true);
    }
    // Set this->device_ and this->context_
    this->device_ = graphlily::find_device();
    this->context_ = cl::Context(this->device_, NULL, NULL, NULL);

    // Set this->kernel_
    auto file_buf = xcl::read_binary_file(xclbin_file_path);
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(this->context_, {this->device_}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    OCL_CHECK(err, this->kernel_ = cl::Kernel(program, this->kernel_name_.c_str(), &err));

    // Set this->command_queue_
    OCL_CHECK(err, this->command_queue_ = cl::CommandQueue(this->context_,
                                                           this->device_,
                                                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                           &err));
    
    this->set_unused_args();// Set unused arguments
    this->set_mode();// Set the mode
}

}  // namespace module
}  // namespace demo

#endif  // DEMO_BASE_MODULE_H_
