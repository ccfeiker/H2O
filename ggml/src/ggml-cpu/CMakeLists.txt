ggml_add_backend_library(ggml-cpu
                         ggml-cpu.c
                         ggml-cpu.cpp
                         ggml-cpu-aarch64.c
                         ggml-cpu-aarch64.h
                         ggml-cpu-quants.c
                         ggml-cpu-quants.h
                        )

target_include_directories(ggml-cpu PRIVATE .)

if (APPLE AND GGML_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        add_compile_definitions(ACCELERATE_NEW_LAPACK)
        add_compile_definitions(ACCELERATE_LAPACK_ILP64)

        target_link_libraries(ggml-cpu PRIVATE ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        message(STATUS "OpenMP found")

        add_compile_definitions(GGML_USE_OPENMP)

        target_link_libraries(ggml-cpu PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)

        # FIXME: should be replaced with a compiler id check
        #if (GGML_MUSA)
        #    list(APPEND GGML_CPU_EXTRA_INCLUDES     "/usr/lib/llvm-14/lib/clang/14.0.0/include")
        #    list(APPEND GGML_CPU_EXTRA_LIBS_PRIVATE "/usr/lib/llvm-14/lib/libomp.so")
        #endif()
    else()
        message(WARNING "OpenMP not found")
    endif()
endif()

if(GGML_OFFLINE_PLANNING_LOG)
    add_compile_definitions(GGML_USE_OFFLINE_PLANNING_LOG)
endif()

if (GGML_LLAMAFILE)
    message(STATUS "Using llamafile")

    add_compile_definitions(GGML_USE_LLAMAFILE)

    target_sources(ggml-cpu PRIVATE
                    llamafile/sgemm.cpp
                    llamafile/sgemm.h)
endif()

if (GGML_CPU_HBM)
    find_library(memkind memkind REQUIRED)

    message(STATUS "Using memkind for CPU HBM")

    add_compile_definitions(GGML_USE_CPU_HBM)

    target_link_libraries(ggml-cpu PUBLIC memkind)
endif()

if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
    CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
    (NOT CMAKE_OSX_ARCHITECTURES      AND
     NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))

    message(STATUS "ARM detected")

    if (MSVC)
        add_compile_definitions(__aarch64__) # MSVC defines _M_ARM64 instead
        add_compile_definitions(__ARM_NEON)
        add_compile_definitions(__ARM_FEATURE_FMA)

        set(CMAKE_REQUIRED_FLAGS_PREV ${CMAKE_REQUIRED_FLAGS})
        string(JOIN " " CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} "/arch:armv8.2")

        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
        if (GGML_COMPILER_SUPPORT_DOTPROD)
            add_compile_definitions(__ARM_FEATURE_DOTPROD)
        endif ()

        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vmlaq_f32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_MATMUL_INT8)

        if (GGML_COMPILER_SUPPORT_MATMUL_INT8)
            add_compile_definitions(__ARM_FEATURE_MATMUL_INT8)
        endif ()

        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16_t _a; float16x8_t _s = vdupq_n_f16(_a); return 0; }" GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
        if (GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
            add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        endif ()

        set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PREV})
    elseif (APPLE)
        if (GGML_NATIVE)
            set(USER_PROVIDED_MARCH FALSE)
            foreach(flag_var IN ITEMS CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_REQUIRED_FLAGS)
                if ("${${flag_var}}" MATCHES "-march=[a-zA-Z0-9+._-]+")
                    set(USER_PROVIDED_MARCH TRUE)
                    break()
                endif()
            endforeach()

            if (NOT USER_PROVIDED_MARCH)
                set(MARCH_FLAGS "-march=armv8.2a")

                check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
                if (GGML_COMPILER_SUPPORT_DOTPROD)
                    set(MARCH_FLAGS "${MARCH_FLAGS}+dotprod")
                    add_compile_definitions(__ARM_FEATURE_DOTPROD)
                endif ()

                set(TEST_I8MM_FLAGS "-march=armv8.2a+i8mm")

                set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${TEST_I8MM_FLAGS}")
                check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_MATMUL_INT8)
                if (GGML_COMPILER_SUPPORT_MATMUL_INT8)
                    set(MARCH_FLAGS "${MARCH_FLAGS}+i8mm")
                    add_compile_definitions(__ARM_FEATURE_MATMUL_INT8)
                endif ()
                set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

                list(APPEND ARCH_FLAGS "${MARCH_FLAGS}")
            endif ()
        endif ()
    else()
        check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if (NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            list(APPEND ARCH_FLAGS -mfp16-format=ieee)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
                # Android armeabi-v7a
                list(APPEND ARCH_FLAGS -mfpu=neon-vfpv4 -mno-unaligned-access -funsafe-math-optimizations)
            else()
                # Raspberry Pi 2
                list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
            endif()
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Android arm64-v8a
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            list(APPEND ARCH_FLAGS -mno-unaligned-access)
        endif()
        if (GGML_SVE)
            list(APPEND ARCH_FLAGS -march=armv8.6-a+sve)
        endif()
    endif()
elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64)$"))
    message(STATUS "x86 detected")
    if (MSVC)
        # instruction set detection for MSVC only
        if (GGML_NATIVE)
            # TODO: improve, should not reference files from the parent folder
            include(cmake/FindSIMD.cmake)
        endif ()
        if (GGML_AVX512)
            list(APPEND ARCH_FLAGS /arch:AVX512)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (GGML_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
                if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                    list(APPEND ARCH_FLAGS -mavx512vbmi)
                endif()
            endif()
            if (GGML_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
                if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                    list(APPEND ARCH_FLAGS -mavx512vnni)
                endif()
            endif()
            if (GGML_AVX512_BF16)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512BF16__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512BF16__>)
                if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                    list(APPEND ARCH_FLAGS -mavx512bf16)
                endif()
            endif()
            if (GGML_AMX_TILE)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AMX_TILE__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AMX_TILE__>)
            endif()
            if (GGML_AMX_INT8)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AMX_INT8__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AMX_INT8__>)
            endif()
            if (GGML_AMX_BF16)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AMX_BF16__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AMX_BF16__>)
            endif()
        elseif (GGML_AVX2)
            list(APPEND ARCH_FLAGS /arch:AVX2)
        elseif (GGML_AVX)
            list(APPEND ARCH_FLAGS /arch:AVX)
        endif()
    else()
        if (GGML_NATIVE)
            list(APPEND ARCH_FLAGS -march=native)
        endif()
        if (GGML_F16C)
            list(APPEND ARCH_FLAGS -mf16c)
        endif()
        if (GGML_FMA)
            list(APPEND ARCH_FLAGS -mfma)
        endif()
        if (GGML_AVX)
            list(APPEND ARCH_FLAGS -mavx)
        endif()
        if (GGML_AVX2)
            list(APPEND ARCH_FLAGS -mavx2)
        endif()
        if (GGML_AVX512)
            list(APPEND ARCH_FLAGS -mavx512f)
            list(APPEND ARCH_FLAGS -mavx512dq)
            list(APPEND ARCH_FLAGS -mavx512bw)
        endif()
        if (GGML_AVX512_VBMI)
            list(APPEND ARCH_FLAGS -mavx512vbmi)
        endif()
        if (GGML_AVX512_VNNI)
            list(APPEND ARCH_FLAGS -mavx512vnni)
        endif()
        if (GGML_AVX512_BF16)
            list(APPEND ARCH_FLAGS -mavx512bf16)
        endif()
        if (GGML_AMX_TILE)
            list(APPEND ARCH_FLAGS -mamx-tile)
        endif()
        if (GGML_AMX_INT8)
            list(APPEND ARCH_FLAGS -mamx-int8)
        endif()
        if (GGML_AMX_BF16)
            list(APPEND ARCH_FLAGS -mamx-bf16)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    execute_process(COMMAND bash -c "grep POWER10 /proc/cpuinfo | head -n 1" OUTPUT_VARIABLE POWER10_M)
    string(FIND "${POWER10_M}" "POWER10" substring_index)
    if (NOT DEFINED substring_index OR "${substring_index}" STREQUAL "")
        set(substring_index -1)
    endif()

    if (${substring_index} GREATER_EQUAL 0)
       list(APPEND ARCH_FLAGS -mcpu=power10)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
       list(APPEND ARCH_FLAGS -mcpu=powerpc64le)
    else()
        list(APPEND ARCH_FLAGS -mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    message(STATUS "loongarch64 detected")

    list(APPEND ARCH_FLAGS -march=loongarch64)
    if (GGML_LASX)
        list(APPEND ARCH_FLAGS -mlasx)
    endif()
    if (GGML_LSX)
        list(APPEND ARCH_FLAGS -mlsx)
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
    message(STATUS "RISC-V detected")
    if (GGML_RVV)
        list(APPEND ARCH_FLAGS -march=rv64gcv -mabi=lp64d)
    endif()
else()
    message(STATUS "Unknown architecture")
endif()

if (GGML_CPU_AARCH64)
    message(STATUS "Using runtime weight conversion of Q4_0 to Q4_0_x_x to enable optimized GEMM/GEMV kernels")
    add_compile_definitions(GGML_USE_CPU_AARCH64)
endif()

target_compile_options(ggml-cpu PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ARCH_FLAGS}>")
target_compile_options(ggml-cpu PRIVATE "$<$<COMPILE_LANGUAGE:C>:${ARCH_FLAGS}>")

if (EMSCRIPTEN)
    set_target_properties(ggml-cpu PROPERTIES COMPILE_FLAGS "-msimd128")
endif()
