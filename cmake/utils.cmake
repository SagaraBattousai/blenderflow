
#Useful for windows where it helps to have dlls in same dir as exe
function(copy_link_libraries_to_build target)
  get_target_property(linked_libraries ${target} LINK_LIBRARIES)

  foreach (library ${linked_libraries})
    list(APPEND target_libraries "$<TARGET_FILE:${library}>")
  endforeach()

  add_custom_command(
    TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E 
      copy_if_different
        "${target_libraries}"
        "$<TARGET_FILE_DIR:${target}>"
    VERBATIM)
endfunction()
