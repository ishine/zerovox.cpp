
add_executable(zerovox
               fs2encoder.cpp
               stylettsdec.cpp
               hifigan.cpp
               utils.cpp
               zerovox.cpp)
target_link_libraries(zerovox PRIVATE ggml)

target_link_libraries(zerovox PRIVATE ${LIBSNDFILE_LIBRARIES}) 

if(MSVC)
  target_compile_options(zerovox PRIVATE /W4 /WX)
else()
  target_compile_options(zerovox PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
