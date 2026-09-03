set(COMMON_SOURCES
    MultilinearModel.cpp
)

set(COMMON_HEADERS
    MultilinearModel.h
    utils.h
)

add_executable(mlmviewer
    main.cpp
    MLMViewer.cpp
    MLMViewer.h
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)
target_link_libraries(mlmviewer pmp_vis)

add_executable(mlm_export
    MLMExport.cpp
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)
target_link_libraries(mlm_export pmp)

add_executable(mlm_fit
    MLMFit.cpp
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)
target_link_libraries(mlm_fit pmp)

if (EMSCRIPTEN)
    set_target_properties(mlmviewer PROPERTIES LINK_FLAGS "--shell-file ${PROJECT_SOURCE_DIR}/external/pmp-library/src/apps/data/shell.html --preload-file ${PROJECT_SOURCE_DIR}/data@../data")
endif()
