if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_sd_1_5 test_sd_1_5.cpp)
    target_link_libraries(test_sd_1_5 PRIVATE stable-diffusion)
    target_compile_features(test_sd_1_5 PRIVATE cxx_std_17)
    add_test(NAME test_sd_1_5 COMMAND test_sd_1_5)

    add_executable(sd-test-unet-perf perf/test_unet.cpp)
    target_link_libraries(sd-test-unet-perf PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
endif()
