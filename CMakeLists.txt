cmake_minimum_required(VERSION 3.18)
project(Atoms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG QUIET)
find_package(GLEW CONFIG QUIET)
find_package(glm CONFIG QUIET)

find_package(PkgConfig QUIET)

set(ATOM_GLFW_TARGET "")
if(TARGET glfw)
    set(ATOM_GLFW_TARGET glfw)
elseif(TARGET glfw3::glfw)
    set(ATOM_GLFW_TARGET glfw3::glfw)
elseif(PkgConfig_FOUND)
    pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
    set(ATOM_GLFW_TARGET PkgConfig::GLFW3)
else()
    message(FATAL_ERROR "GLFW3 not found. Install glfw or provide it with vcpkg.")
endif()

set(ATOM_GLEW_TARGET "")
if(TARGET GLEW::GLEW)
    set(ATOM_GLEW_TARGET GLEW::GLEW)
elseif(PkgConfig_FOUND)
    pkg_check_modules(GLEW REQUIRED IMPORTED_TARGET glew)
    set(ATOM_GLEW_TARGET PkgConfig::GLEW)
else()
    message(FATAL_ERROR "GLEW not found. Install glew or provide it with vcpkg.")
endif()

set(ATOM_GLM_TARGET "")
if(TARGET glm::glm)
    set(ATOM_GLM_TARGET glm::glm)
elseif(PkgConfig_FOUND)
    pkg_check_modules(GLM REQUIRED IMPORTED_TARGET glm)
    set(ATOM_GLM_TARGET PkgConfig::GLM)
else()
    message(FATAL_ERROR "GLM not found. Install glm or provide it with vcpkg.")
endif()

add_library(atom_common INTERFACE)
target_link_libraries(atom_common
    INTERFACE
    OpenGL::GL
    ${ATOM_GLFW_TARGET}
    ${ATOM_GLEW_TARGET}
    ${ATOM_GLM_TARGET}
)

if(APPLE)
    target_compile_definitions(atom_common INTERFACE GL_SILENCE_DEPRECATION)
endif()

function(add_atom_target target source_file)
    add_executable(${target} ${source_file})
    target_link_libraries(${target} PRIVATE atom_common)
endfunction()

add_atom_target(atom src/atom.cpp)
add_atom_target(wave_atom_2d src/wave_atom_2d.cpp)
add_atom_target(atom_realtime src/atom_realtime.cpp)

if(APPLE)
    message(STATUS "Skipping atom_raytracer on macOS (requires OpenGL 4.3 + SSBO, macOS supports up to 4.1).")
else()
    add_atom_target(atom_raytracer src/atom_raytracer.cpp)
endif()
