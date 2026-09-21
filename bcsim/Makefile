# Builds the simulator libraries that bcsim/env.py and the tests load via ctypes.
#
#     make -C bcsim
#
# Each library is built to a temp file and then moved into place: a live run
# has the old one mmapped, and overwriting it in place would crash that run.

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -shared -fPIC -pthread
OUT      := bcsim

all: $(OUT)/libbcvec.so $(OUT)/libbctext.so

$(OUT)/libbcvec.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbctext.so: cpp/bc_capi_text.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) $< -o $@.tmp && mv $@.tmp $@

clean:
	rm -f $(OUT)/libbcvec.so $(OUT)/libbctext.so

.PHONY: all clean
