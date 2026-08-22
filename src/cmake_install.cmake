# Install script for directory: /global/homes/y/yufeid/workspace/changchen/andrewy/flux/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/python/flux")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/comm_none/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/coll/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/ag_gemm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/gemm_rs/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/gemm_a2a_transpose/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/a2a_transpose_gemm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/inplace_cast/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/quantization/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/moe_ag_scatter/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/moe_gather_rs/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/ths_op/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/flux/" TYPE DIRECTORY FILES
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/src/./"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/"
    FILES_MATCHING REGEX "/[^/]*\\/ths\\_op\\/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/flux" TYPE FILE FILES
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/a2av_progress.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/flux.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/gemm_hparams.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/gemm_meta.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/gemm_operator_base.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/op_registry.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/op_registry_proto_utils.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/runtime_config.h"
    "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/include/flux/utils.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Flux" TYPE FILE FILES "/global/homes/y/yufeid/workspace/changchen/andrewy/flux/cmake/FluxConfig.cmake")
endif()

