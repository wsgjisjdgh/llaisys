#include "op.hpp"
#include "cpu/rope_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    auto device = core::context().runtime().deviceType();

    if (device == LLAISYS_DEVICE_CPU) {

        if (in->ndim() != 3) {
            throw std::invalid_argument("RoPE: Input tensor must be 3D [seq, head, dim]");
        }
        
        size_t seq_len = in->shape()[0];
        size_t n_heads = in->shape()[1];
        size_t head_dim = in->shape()[2];

        if (head_dim % 2 != 0) {
            throw std::invalid_argument("RoPE: Head dimension must be even");
        }
        
        // 检查 pos_ids
        if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
            throw std::invalid_argument("RoPE: pos_ids must be int64");
        }
        if (pos_ids->numel() != seq_len) {
             throw std::invalid_argument("RoPE: pos_ids length mismatch");
        }

        cpu::rope(out->data(), in->data(), pos_ids->data(),
                  in->dtype(), seq_len, n_heads, head_dim, theta);
    } else {
        throw std::runtime_error("RoPE: Unsupported device type");
    }
}

} // namespace llaisys::ops