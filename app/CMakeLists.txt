add_executable(app app.c)

file(COPY ${ASSETS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(app PUBLIC model)

copy_link_libraries_to_build(app)
