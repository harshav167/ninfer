target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/clipped_direction_removal.cu"
  "${CMAKE_CURRENT_LIST_DIR}/l2norm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/layer_norm.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm/sources.cmake")
