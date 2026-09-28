target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/distributed_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

target_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4_non_rdc.cu"
)

# FP8 attention and its cache-append codec support the secondary Ampere device.
# Keep model execution and the remaining Ops compiled for the primary SM120 target.
add_library(ninfer_fp8_attention OBJECT
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../kv_cache/append/launch.cu")
ninfer_internal_includes(ninfer_fp8_attention)
ninfer_cuda_non_rdc_archive(ninfer_fp8_attention)
set_target_properties(ninfer_fp8_attention PROPERTIES CUDA_ARCHITECTURES "86;120a")
target_link_libraries(ninfer_fp8_attention PRIVATE CUDA::cudart)
target_sources(ninfer_ops PRIVATE $<TARGET_OBJECTS:ninfer_fp8_attention>)
