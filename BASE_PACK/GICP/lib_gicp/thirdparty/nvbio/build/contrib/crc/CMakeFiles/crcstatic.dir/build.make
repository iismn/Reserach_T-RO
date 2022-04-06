# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build

# Include any dependencies generated for this target.
include contrib/crc/CMakeFiles/crcstatic.dir/depend.make

# Include the progress variables for this target.
include contrib/crc/CMakeFiles/crcstatic.dir/progress.make

# Include the compile flags for this target's objects.
include contrib/crc/CMakeFiles/crcstatic.dir/flags.make

contrib/crc/CMakeFiles/crcstatic.dir/crc.o: contrib/crc/CMakeFiles/crcstatic.dir/flags.make
contrib/crc/CMakeFiles/crcstatic.dir/crc.o: ../contrib/crc/crc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object contrib/crc/CMakeFiles/crcstatic.dir/crc.o"
	cd /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/crcstatic.dir/crc.o -c /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/contrib/crc/crc.cpp

contrib/crc/CMakeFiles/crcstatic.dir/crc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/crcstatic.dir/crc.i"
	cd /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/contrib/crc/crc.cpp > CMakeFiles/crcstatic.dir/crc.i

contrib/crc/CMakeFiles/crcstatic.dir/crc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/crcstatic.dir/crc.s"
	cd /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/contrib/crc/crc.cpp -o CMakeFiles/crcstatic.dir/crc.s

# Object files for target crcstatic
crcstatic_OBJECTS = \
"CMakeFiles/crcstatic.dir/crc.o"

# External object files for target crcstatic
crcstatic_EXTERNAL_OBJECTS =

contrib/crc/libcrcstatic.a: contrib/crc/CMakeFiles/crcstatic.dir/crc.o
contrib/crc/libcrcstatic.a: contrib/crc/CMakeFiles/crcstatic.dir/build.make
contrib/crc/libcrcstatic.a: contrib/crc/CMakeFiles/crcstatic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcrcstatic.a"
	cd /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc && $(CMAKE_COMMAND) -P CMakeFiles/crcstatic.dir/cmake_clean_target.cmake
	cd /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/crcstatic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
contrib/crc/CMakeFiles/crcstatic.dir/build: contrib/crc/libcrcstatic.a

.PHONY : contrib/crc/CMakeFiles/crcstatic.dir/build

contrib/crc/CMakeFiles/crcstatic.dir/clean:
	cd /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc && $(CMAKE_COMMAND) -P CMakeFiles/crcstatic.dir/cmake_clean.cmake
.PHONY : contrib/crc/CMakeFiles/crcstatic.dir/clean

contrib/crc/CMakeFiles/crcstatic.dir/depend:
	cd /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/contrib/crc /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc /home/root/Workspace-IISMN/ROS-BASE/NvBIO/nvbio/build/contrib/crc/CMakeFiles/crcstatic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : contrib/crc/CMakeFiles/crcstatic.dir/depend

