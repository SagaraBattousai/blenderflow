
function(dev_python_extension_build target)
  if (${ARGC} GREATER 1)
    # set(base_dir ${ARGV1})
    get_filename_component(base_dir ${ARGV1} ABSOLUTE)
  else()
    set(in_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    get_filename_component(base_dir ${in_dir}/../.. ABSOLUTE)
  endif()

  add_custom_command(
    TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E 
      copy_if_different
        "$<TARGET_FILE:${target}>"
        "${base_dir}/$<TARGET_FILE_BASE_NAME:${target}>${PYTHON_EXTENSION}"
    COMMAND ${CMAKE_COMMAND} -E 
      echo 
        "Copying ${target} to ${base_dir} for dev build"
    VERBATIM)
endfunction()


set(BUILD_PYTHON_EXTENSIONS ON CACHE BOOL 
"Whether to build Python C Extension Libraries and set the appropriate \
extension") 

set(PYTHON_EXTENSION_DOC_STRING
"Do not manually change this variable unless you are sure you need/want to. \
Using setup.py with CMake_Extensions will set this automatically and running \
CMake with variables left as default will set the correct extension for \
libraries without the python version information that setup.py includes.")

if (BUILD_PYTHON_EXTENSIONS AND (NOT PYTHON_EXTENSION))
  if (${WIN32})
    set(PYTHON_EXTENSION ".pyd" CACHE STRING ${PYTHON_EXTENSION_DOC_STRING})
  else()
    set(PYTHON_EXTENSION ".so"  CACHE STRING ${PYTHON_EXTENSION_DOC_STRING})
  endif()
endif()

set(BUILD_EXTENSIONS_FOR_DEVELOPMENT OFF CACHE BOOL 
  "Whether to copy the built Python C Extension Libraries to the source \
  directory the module expects if running as uninstalled package \
  (useful for running tests on CI, not requiring pip install, \
  allowing use of pip install -e . and for tasks such as running \
  tests etc)")
