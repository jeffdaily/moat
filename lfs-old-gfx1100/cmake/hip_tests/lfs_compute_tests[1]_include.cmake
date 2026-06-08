if(EXISTS "/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests")
  if(NOT EXISTS "/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests[1]_tests.cmake" OR
     NOT "/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests[1]_tests.cmake" IS_NEWER_THAN "/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests" OR
     NOT "/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests[1]_tests.cmake" IS_NEWER_THAN "${CMAKE_CURRENT_LIST_FILE}")
    include("/opt/conda/envs/py_3.12/lib/python3.12/site-packages/cmake/data/share/cmake-4.0/Modules/GoogleTestAddTests.cmake")
    gtest_discover_tests_impl(
      TEST_EXECUTABLE [==[/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests]==]
      TEST_EXECUTOR [==[]==]
      TEST_WORKING_DIR [==[/var/lib/jenkins/moat/lfs-old-gfx1100]==]
      TEST_EXTRA_ARGS [==[]==]
      TEST_PROPERTIES [==[TIMEOUT;120]==]
      TEST_PREFIX [==[]==]
      TEST_SUFFIX [==[]==]
      TEST_FILTER [==[]==]
      NO_PRETTY_TYPES [==[FALSE]==]
      NO_PRETTY_VALUES [==[FALSE]==]
      TEST_LIST [==[lfs_compute_tests_TESTS]==]
      CTEST_FILE [==[/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests[1]_tests.cmake]==]
      TEST_DISCOVERY_TIMEOUT [==[5]==]
      TEST_DISCOVERY_EXTRA_ARGS [==[]==]
      TEST_XML_OUTPUT_DIR [==[]==]
    )
  endif()
  include("/var/lib/jenkins/moat/lfs-old-gfx1100/cmake/hip_tests/lfs_compute_tests[1]_tests.cmake")
else()
  add_test(lfs_compute_tests_NOT_BUILT lfs_compute_tests_NOT_BUILT)
endif()
