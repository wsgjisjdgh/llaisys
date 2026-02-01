#include "op.hpp"
#include "cpu/argmax_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    auto device = core::context().runtime().deviceType();

    if (device == LLAISYS_DEVICE_CPU) {
        cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    } else {
        throw std::runtime_error("Argmax: Unsupported device type");
    }
}

} // namespace llaisys::ops