#!/bin/bash

# 设置构建目录
BUILD_DIR="build"

# 如果构建目录已存在，则删除它
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

# 创建新的构建目录
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_CXX_FLAGS="-march=native -O3" ..

cmake --build . --config Release