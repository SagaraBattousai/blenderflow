set(src "vec.c" "tri.c")

set(header_dir "Primatives")
set(pubic_headers "vec.h" "tri.h")

list(TRANSFORM pubic_headers PREPEND "${INCLUDE_DIR}/${header_dir}/")

add_library(primatives STATIC ${src} ${pubic_headers})

target_include_directories(primatives PUBLIC
  $<BUILD_INTERFACE:${INCLUDE_DIR}>
  )
