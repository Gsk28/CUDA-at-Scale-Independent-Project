.PHONY: configure build demo clean

BUILD_DIR ?= build
BUILD_TYPE ?= Release

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j

demo:
	bash ./scripts/run_demo.sh

clean:
	cmake -E rm -rf $(BUILD_DIR) data/input data/output proof/latest
