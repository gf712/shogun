# taken from https://github.com/jfultz/libtooling_step_by_step/blob/master/CMakeLists.txt
cmake_minimum_required (VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -fno-strict-aliasing -fno-exceptions -fno-rtti")

project (test)

set(LIBRARY_LIST clangFrontend  clangSerialization clangDriver clangParse clangRewriteFrontend clangStaticAnalyzerFrontend clangSema)
set(LIBRARY_LIST ${LIBRARY_LIST} clangAnalysis clangEdit clangAST clangLex clangBasic clangTooling
      clangRewrite clangASTMatchers clangToolingCore)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLVM include directory: ${LLVM_INCLUDE_DIRS}")
message(STATUS "Using LLVM library directory: ${LLVM_LIBRARY_DIRS}")

link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_executable(rename_class_members rename_class_members.cpp)
target_link_libraries(rename_class_members ${LIBRARY_LIST})

target_link_libraries(rename_class_members
	LLVMX86AsmParser # MC, MCParser, Support, X86Desc, X86Info
	LLVMX86Desc # MC, Support, X86AsmPrinter, X86Info
	LLVMX86AsmPrinter # MC, Support, X86Utils
	LLVMX86Info # MC, Support, Target
	LLVMX86Utils # Core, Support
	LLVMipo
	LLVMScalarOpts
	LLVMInstCombine
	LLVMTransformUtils
	LLVMAnalysis
	LLVMTarget
	LLVMOption # Support
	LLVMMCParser # MC, Support
	LLVMMC # Object, Support
	LLVMObject # BitReader, Core, Support
	LLVMBitReader # Core, Support
	LLVMCore # Support
	LLVMSupport
)