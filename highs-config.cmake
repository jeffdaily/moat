

set(HIGHS_DIR "")

if (FAST_BUILD)
  if(NOT TARGET highs)
    include("${CMAKE_CURRENT_LIST_DIR}/highs-targets.cmake")
  endif()

  set(HIGHS_LIBRARIES highs)
else()
  if(NOT TARGET libhighs)
    include("${CMAKE_CURRENT_LIST_DIR}/highs-targets.cmake")
  endif()

  set(HIGHS_LIBRARIES libhighs)
endif() 

set(HIGHS_INCLUDE_DIRS "/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/src;/var/lib/jenkins/moat")

set(HIGHS_FOUND TRUE)

include(CMakeFindDependencyMacro)
find_dependency(ZLIB)
