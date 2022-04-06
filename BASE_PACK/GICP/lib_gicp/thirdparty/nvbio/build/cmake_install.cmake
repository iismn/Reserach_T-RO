# Install script for directory: /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
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
  set(CMAKE_INSTALL_SO_NO_EXE "1")
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
  # Include the install script for each subdirectory.
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/zlib-1.2.7/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/lz4/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/bamtools/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/moderngpu/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/htslib/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvbio/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvbio-test/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvBowtie/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvFM-server/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvBWT/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvSetBWT/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvSSA/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvExtractReads/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvLighter/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvmem/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvbio-aln-diff/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/nvMicroAssembly/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/sufsort-test/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/sw-benchmark/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/examples/waveletfm/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/examples/proteinsw/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/examples/seeding/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/examples/fmmap/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/examples/qmap/cmake_install.cmake")
  include("/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/examples/mem/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
