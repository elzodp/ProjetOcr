# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/elhadji/Bureau/ProjetOCR

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/elhadji/Bureau/ProjetOCR/src

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/elhadji/Bureau/ProjetOCR/src/CMakeFiles /home/elhadji/Bureau/ProjetOCR/src/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/elhadji/Bureau/ProjetOCR/src/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named OCR

# Build rule for target.
OCR: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 OCR
.PHONY : OCR

# fast build rule for target.
OCR/fast:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/build
.PHONY : OCR/fast

NN2.o: NN2.cpp.o

.PHONY : NN2.o

# target to build an object file
NN2.cpp.o:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/NN2.cpp.o
.PHONY : NN2.cpp.o

NN2.i: NN2.cpp.i

.PHONY : NN2.i

# target to preprocess a source file
NN2.cpp.i:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/NN2.cpp.i
.PHONY : NN2.cpp.i

NN2.s: NN2.cpp.s

.PHONY : NN2.s

# target to generate assembly for a file
NN2.cpp.s:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/NN2.cpp.s
.PHONY : NN2.cpp.s

Timer.o: Timer.cpp.o

.PHONY : Timer.o

# target to build an object file
Timer.cpp.o:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/Timer.cpp.o
.PHONY : Timer.cpp.o

Timer.i: Timer.cpp.i

.PHONY : Timer.i

# target to preprocess a source file
Timer.cpp.i:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/Timer.cpp.i
.PHONY : Timer.cpp.i

Timer.s: Timer.cpp.s

.PHONY : Timer.s

# target to generate assembly for a file
Timer.cpp.s:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/Timer.cpp.s
.PHONY : Timer.cpp.s

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/main.cpp.s
.PHONY : main.cpp.s

tests/test_xor.o: tests/test_xor.cpp.o

.PHONY : tests/test_xor.o

# target to build an object file
tests/test_xor.cpp.o:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/tests/test_xor.cpp.o
.PHONY : tests/test_xor.cpp.o

tests/test_xor.i: tests/test_xor.cpp.i

.PHONY : tests/test_xor.i

# target to preprocess a source file
tests/test_xor.cpp.i:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/tests/test_xor.cpp.i
.PHONY : tests/test_xor.cpp.i

tests/test_xor.s: tests/test_xor.cpp.s

.PHONY : tests/test_xor.s

# target to generate assembly for a file
tests/test_xor.cpp.s:
	$(MAKE) -f CMakeFiles/OCR.dir/build.make CMakeFiles/OCR.dir/tests/test_xor.cpp.s
.PHONY : tests/test_xor.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... OCR"
	@echo "... NN2.o"
	@echo "... NN2.i"
	@echo "... NN2.s"
	@echo "... Timer.o"
	@echo "... Timer.i"
	@echo "... Timer.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... tests/test_xor.o"
	@echo "... tests/test_xor.i"
	@echo "... tests/test_xor.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

