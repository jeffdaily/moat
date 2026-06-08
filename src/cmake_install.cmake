# Install script for directory: /var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/install")
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

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/rocm/llvm/bin/llvm-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/filereaderlp" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/../extern/filereaderlp/builder.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/filereaderlp" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/../extern/filereaderlp/model.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/filereaderlp" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/../extern/filereaderlp/reader.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/Filereader.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/FilereaderLp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/FilereaderEms.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/FilereaderMps.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/HMpsFF.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/HMPSIO.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/HighsIO.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/io/LoadOptions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HConst.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HStruct.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsAnalysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsCallback.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsCallbackStruct.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsInfo.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsInfoDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsLp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsLpSolverObject.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsLpUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsModelUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsOptions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsRanging.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsRuntimeOptions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsSolution.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsSolutionDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsSolve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/lp_data/HighsStatus.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsCliqueTable.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsCutGeneration.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsConflictPool.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsCutPool.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsDebugSol.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsDomainChange.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsDomain.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsDynamicRowMatrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsGFkSolve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsImplications.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsLpAggregator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsLpRelaxation.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsMipSolverData.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsMipSolver.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsModkSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsNodeQueue.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsObjectiveFunction.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsPathSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsPrimalHeuristics.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsPseudocost.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsRedcostFixing.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsSearch.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsSeparation.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsTableauSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/mip/HighsTransformedLp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/model" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/model/HighsHessian.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/model" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/model/HighsHessianUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/model" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/model/HighsModel.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsBinarySemaphore.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsCacheAlign.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsCombinable.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsMutex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsParallel.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsRaceTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsSchedulerConstants.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsSpinMutex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsSplitDeque.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsTaskExecutor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/parallel/HighsTask.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/qpsolver/a_asm.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/qpsolver/a_quass.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/qpsolver/quass.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/qpsolver/vector.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/qpsolver/scaling.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/qpsolver/perturbation.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HApp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HEkk.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HEkkDual.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HEkkDualRHS.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HEkkDualRow.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HEkkPrimal.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HighsSimplexAnalysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HSimplex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HSimplexReport.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HSimplexDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/HSimplexNla.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/SimplexConst.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/SimplexStruct.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/simplex/SimplexTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/ICrash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/ICrashUtil.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/ICrashX.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/HighsPostsolveStack.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/HighsSymmetry.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/HPresolve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/HPresolveAnalysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/presolve/PresolveComponent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/test" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/test/DevKkt.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/test" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/test/KktCh2.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/FactorTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HFactor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HFactorConst.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HFactorDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsCDouble.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsComponent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsDataStack.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsDisjointSets.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsHash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsHashTree.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsInt.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsIntegers.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsLinearSumBounds.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsMatrixPic.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsMatrixSlice.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsMatrixUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsRandom.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsRbTree.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsSort.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsSparseMatrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsSparseVectorSum.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsSplay.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HighsUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HSet.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HVector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/HVectorBase.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/util/stringutil.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/Highs.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/interfaces" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/interfaces/highs_c_api.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm" TYPE FILE FILES "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src/ipm/IpxWrapper.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs" TYPE FILE FILES "/var/lib/jenkins/moat/HConfig.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/highs" TYPE FILE FILES "/var/lib/jenkins/moat/CMakeFiles/highs-config.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/var/lib/jenkins/moat/CMakeFiles/highs.pc")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/var/lib/jenkins/moat/src/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
