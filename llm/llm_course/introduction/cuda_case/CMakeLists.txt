cmake_minimum_required(VERSION 3.2)
project(mmul_cpp_cu C CXX)

# find correct version of Python
execute_process(COMMAND python3.8-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
  else()
    set(CMAKE_CXX_FLAGS "/std:c++11 -O2 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
  endif()
else()
  if(NOT MSVC)
      set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
      set(CMAKE_CUDA_STANDARD 14)
    else()
      set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
      set(CMAKE_CUDA_STANDARD 14)
    endif()
endif()
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
####### CPU #######
###################
add_library(mmul_cpp MODULE ./src/mmul.cc)
target_link_libraries(mmul_cpp PUBLIC ${LINKER_LIBS})
pybind11_extension(mmul_cpp)
pybind11_strip(mmul_cpp)


set_target_properties(mmul_cpp
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mmul_cpp_cu
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET mmul_cpp PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()


###################
####### GPU #######
###################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(mmul_cuda MODULE src/mmul.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(mmul_cuda ${LINKER_LIBS})
  pybind11_extension(mmul_cuda)
  pybind11_strip(mmul_cuda)

  # directly output to ffi folder
  set_target_properties(mmul_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mmul_cpp_cu
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()