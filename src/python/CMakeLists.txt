
set(src "blenderflow.c" "arrayutils.c")

set(header_dir "Binding/Python")
set(public_headers "model.h" "arrayutils.h")

list(TRANSFORM public_headers PREPEND "${INCLUDE_DIR}/${header_dir}/")

#Headers are added so Visual studios sln looks nice
add_library(blenderflow
  SHARED
  ${src}
  ${public_headers})


target_include_directories(blenderflow
  PUBLIC 
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${INCLUDE_DIR}>
  )

message(STATUS ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

#Important for Numpy extensins
target_compile_definitions(blenderflow PUBLIC 
  MODULE_ARRAY_API_NAME=blenderflow_ARRAY_API)


target_link_libraries(blenderflow 
  PUBLIC 
  model 
  endianness
  ${Python3_LIBRARIES}
  ${Python3_NumPy_LIBRARIES})
# target_link_libraries(blenderflow PRIVATE ${Python3_LIBRARIES})

if (PYTHON_EXTENSION)
  message(DEBUG "Setting library extension to ${PYTHON_EXTENSION}")

  set_target_properties(blenderflow 
    PROPERTIES 
    SUFFIX ${PYTHON_EXTENSION}
    PREFIX "")
endif()

if(BUILD_EXTENSIONS_FOR_DEVELOPMENT)
  message(STATUS "Build will copy C Extension inplace")
  dev_python_extension_build(blenderflow "${CMAKE_SOURCE_DIR}")
endif()


