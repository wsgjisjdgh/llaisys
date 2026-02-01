#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    auto device = core::context().runtime().deviceType();

    if (device == LLAISYS_DEVICE_CPU) {
        size_t numel = out->numel();
        if (gate->numel() != numel || up->numel() != numel) {
            throw std::invalid_argument("SwiGLU: Input/Output tensor shapes mismatch");
        }

        cpu::swiglu(out->data(), gate->data(), up->data(),
                    out->dtype(), numel);
    } else {
        throw std::runtime_error("SwiGLU: Unsupported device type");
    }
}

} // namespace llaisys::ops