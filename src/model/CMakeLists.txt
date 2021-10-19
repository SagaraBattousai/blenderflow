set(src "model_reader.c" "ply_reader.c" "stl_reader.c")

set(header_dir "Model")
set(pubic_headers "model_reader.h" "ply_reader.h" "stl_reader.h")

list(TRANSFORM pubic_headers PREPEND "${INCLUDE_DIR}/${header_dir}/")

add_library(model SHARED ${src} ${pubic_headers})

target_include_directories(model PUBLIC
  $<BUILD_INTERFACE:${INCLUDE_DIR}>
  )

target_link_libraries(model PRIVATE endianness)
target_link_libraries(model PUBLIC primatives)
