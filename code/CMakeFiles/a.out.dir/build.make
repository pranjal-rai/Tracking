# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/gazzib/Downloads/honours/project 2/code"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/gazzib/Downloads/honours/project 2/code"

# Include any dependencies generated for this target.
include CMakeFiles/a.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/a.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/a.out.dir/flags.make

CMakeFiles/a.out.dir/run.cpp.o: CMakeFiles/a.out.dir/flags.make
CMakeFiles/a.out.dir/run.cpp.o: run.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/gazzib/Downloads/honours/project 2/code/CMakeFiles" $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/a.out.dir/run.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/a.out.dir/run.cpp.o -c "/home/gazzib/Downloads/honours/project 2/code/run.cpp"

CMakeFiles/a.out.dir/run.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a.out.dir/run.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/gazzib/Downloads/honours/project 2/code/run.cpp" > CMakeFiles/a.out.dir/run.cpp.i

CMakeFiles/a.out.dir/run.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a.out.dir/run.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/gazzib/Downloads/honours/project 2/code/run.cpp" -o CMakeFiles/a.out.dir/run.cpp.s

CMakeFiles/a.out.dir/run.cpp.o.requires:
.PHONY : CMakeFiles/a.out.dir/run.cpp.o.requires

CMakeFiles/a.out.dir/run.cpp.o.provides: CMakeFiles/a.out.dir/run.cpp.o.requires
	$(MAKE) -f CMakeFiles/a.out.dir/build.make CMakeFiles/a.out.dir/run.cpp.o.provides.build
.PHONY : CMakeFiles/a.out.dir/run.cpp.o.provides

CMakeFiles/a.out.dir/run.cpp.o.provides.build: CMakeFiles/a.out.dir/run.cpp.o

# Object files for target a.out
a_out_OBJECTS = \
"CMakeFiles/a.out.dir/run.cpp.o"

# External object files for target a.out
a_out_EXTERNAL_OBJECTS =

a.out: CMakeFiles/a.out.dir/run.cpp.o
a.out: CMakeFiles/a.out.dir/build.make
a.out: /usr/local/lib/libopencv_core.so.2.4.10
a.out: /usr/local/lib/libopencv_ml.so.2.4.10
a.out: /usr/local/lib/libopencv_video.so.2.4.10
a.out: /usr/local/lib/libopencv_calib3d.so.2.4.10
a.out: /usr/local/lib/libopencv_contrib.so.2.4.10
a.out: /usr/local/lib/libopencv_features2d.so.2.4.10
a.out: /usr/local/lib/libopencv_flann.so.2.4.10
a.out: /usr/local/lib/libopencv_gpu.so.2.4.10
a.out: /usr/local/lib/libopencv_highgui.so.2.4.10
a.out: /usr/local/lib/libopencv_imgproc.so.2.4.10
a.out: /usr/local/lib/libopencv_objdetect.so.2.4.10
a.out: /usr/local/lib/libopencv_legacy.so.2.4.10
a.out: CMakeFiles/a.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable a.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/a.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/a.out.dir/build: a.out
.PHONY : CMakeFiles/a.out.dir/build

CMakeFiles/a.out.dir/requires: CMakeFiles/a.out.dir/run.cpp.o.requires
.PHONY : CMakeFiles/a.out.dir/requires

CMakeFiles/a.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/a.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/a.out.dir/clean

CMakeFiles/a.out.dir/depend:
	cd "/home/gazzib/Downloads/honours/project 2/code" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/gazzib/Downloads/honours/project 2/code" "/home/gazzib/Downloads/honours/project 2/code" "/home/gazzib/Downloads/honours/project 2/code" "/home/gazzib/Downloads/honours/project 2/code" "/home/gazzib/Downloads/honours/project 2/code/CMakeFiles/a.out.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/a.out.dir/depend
