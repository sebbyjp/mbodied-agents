#!/bin/bash

# Function to check the operating system
check_os() {
    case "$(uname)" in
        "Linux")
            echo "Linux"
            ;;
        "Darwin")
            echo "macOS"
            ;;
        "CYGWIN"*|"MINGW"*|"MSYS"*)
            echo "Windows"
            ;;
        *)
           echo "Unknown"
            ;;
    esac
}

# Call the function
if [ "$(check_os)" == "Linux" ]; then
    echo "Linux"
elif [ "$(check_os)" == "macOS" ]; then
    git clone https://github.com/IntelRealSense/librealsense.git
    export CPLUS_INCLUDE_PATH=/usr/local/include/librealsense2:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

    cd librealsense

    mkdir build && cd build
    cmake .. -DBUILD_EXAMPLES=true
    make
    sudo make install
    cd ..
    rm -rf librealsense
    pip install pyrealsense2
    echo "realsense installed successfully on macOS"
elif [ "$(check_os)" == "Windows" ]; then
    echo "Windows"
else
    echo "$(check_os) function failed"
fi