#include "op.hpp"
#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
   auto device=core::context().runtime().deviceType();
   
   if(index->dtype()!=LLAISYS_DTYPE_I64){
       throw std::invalid_argument("Index tensor must be of type INT64.");
   }

   if(device==LLAISYS_DEVICE_CPU){
    size_t num_indices=index->numel();
    size_t embedding_dim=weight->shape()[1];
    cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), num_indices, embedding_dim);
   }
    else{
         throw std::runtime_error("Embedding: Unsupported device type");
}
}
} // namespace llaisys::ops
