
set(src "model.c")

set(header_dir "Binding/Python")
set(public_headers "model.h")

list(TRANSFORM public_headers PREPEND "${INCLUDE_DIR}/${header_dir}/")

#Headers are added so Visual studios sln looks nice
add_library(modelmodule
  SHARED
  ${src}
  ${public_headers})


target_include_directories(modelmodule
  PUBLIC 
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${INCLUDE_DIR}>
  )

message(STATUS ${Python3_INCLUDE_DIRS})

#Remember how this works
target_compile_definitions(modelmodule PUBLIC 
  MODULE_ARRAY_API_NAME=modelmodule_ARRAY_API)


target_link_libraries(modelmodule PUBLIC model)
target_link_libraries(modelmodule PRIVATE ${Python3_LIBRARIES})

if (PYTHON_EXTENSION)
  message(DEBUG "Setting library extension to ${PYTHON_EXTENSION}")

  set_target_properties(modelmodule 
    PROPERTIES 
    SUFFIX ${PYTHON_EXTENSION}
    PREFIX "")
endif()

if(BUILD_EXTENSIONS_INPLACE)
  message(STATUS "Build will copy C Extension inplace")
  inplace_build(modelmodule)
endif()


