set(src "endianness.c")

set(header_dir "Endianness")
set(pubic_headers "endianness.h")

list(TRANSFORM pubic_headers PREPEND "${INCLUDE_DIR}/${header_dir}/")
add_library(endianness STATIC ${src} ${pubic_headers})

target_include_directories(endianness PUBLIC
  $<BUILD_INTERFACE:${INCLUDE_DIR}>
  )




