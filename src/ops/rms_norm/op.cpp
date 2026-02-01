#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    auto device = core::context().runtime().deviceType();

    if (device == LLAISYS_DEVICE_CPU) {

        size_t rows = in->shape()[0];
        size_t cols = in->shape()[1];

        if (weight->numel() != cols) {
             throw std::invalid_argument("RMSNorm: Weight shape mismatch");
        }

        cpu::rms_norm(out->data(), in->data(), weight->data(),
                      in->dtype(), rows, cols, eps);
    } else {
        throw std::runtime_error("RMSNorm: Unsupported device type");
    }
}

} // namespace llaisys::ops