# Builds the simulator libraries that bcsim/env.py and the tests load via ctypes.
#
#     make -C bcsim
#
# Each library is built to a temp file and then moved into place: a live run
# has the old one mmapped, and overwriting it in place would crash that run.

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -shared -fPIC -pthread
OUT      := bcsim

all: $(OUT)/libbcvec.so $(OUT)/libbctext.so $(OUT)/libbcvec_replay.so $(OUT)/libbcvec_priv.so

# sonar v2 (cpp/bc_sonar2.hpp, 2026-09-30): the team packet, its memory merge and the
# grid's five report planes (43 channels). Separate files, so nothing built without it
# changes: `make -C bcsim s2`
s2: $(OUT)/libbcvec_s2.so $(OUT)/libbcvec_priv_s2.so $(OUT)/libbcvec_replay_s2.so
# the next generation: 15x15 grid (BC_GRID_G=15: 7 cells every way, symmetric about the 7x7 window),
# sonar v2, and the len-2 / len-3 splits (BC_XSPLIT: 51 actions)
s2g15: $(OUT)/libbcvec_s2_g15.so $(OUT)/libbcvec_priv_s2_g15.so $(OUT)/libbcvec_replay_s2_g15.so
# s2g15 with the portal report (BC_PORTALREP, 2026-10-02): the packet's probability fields and grid
# channels 39-42 carry what lies through portals instead; and the far sprints (BC_FARSPRINT, 2026-10-02:
# 24 more actions, every window tile reachable in one move: 75 actions). New files, so a run on s2g15
# is untouched
s2g15p: $(OUT)/libbcvec_s2_g15p.so $(OUT)/libbcvec_priv_s2_g15p.so $(OUT)/libbcvec_replay_s2_g15p.so
# 17x17 grid (BC_GRID_G=17), sonar v2: for the grid-size experiment (2026-10-01)
s2g17: $(OUT)/libbcvec_s2_g17.so $(OUT)/libbcvec_priv_s2_g17.so $(OUT)/libbcvec_replay_s2_g17.so

$(OUT)/libbcvec.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) $< -o $@.tmp && mv $@.tmp $@

# for train/replay_dataset.py: real games carry sprints longer than 8 steps
$(OUT)/libbcvec_replay.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

# for finetune.py --privileged and critic studies: the privileged-feature and
# board buffers (bound only on request), and long sprints for replays
$(OUT)/libbcvec_priv.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_s2.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_priv_s2.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_replay_s2.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbctext.so: cpp/bc_capi_text.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) $< -o $@.tmp && mv $@.tmp $@

clean:
	rm -f $(OUT)/libbcvec.so $(OUT)/libbctext.so $(OUT)/libbcvec_replay.so $(OUT)/libbcvec_priv.so

.PHONY: all clean s2 s2g15 s2g15p s2g17

$(OUT)/libbcvec_s2_g17.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_GRID_G=17 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_priv_s2_g17.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_GRID_G=17 -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_replay_s2_g17.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_GRID_G=17 -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_s2_g15.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_XSPLIT -DBC_GRID_G=15 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_priv_s2_g15.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_XSPLIT -DBC_GRID_G=15 -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_replay_s2_g15.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_XSPLIT -DBC_GRID_G=15 -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_s2_g15p.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_XSPLIT -DBC_GRID_G=15 -DBC_PORTALREP -DBC_FARSPRINT -DBC_AHIST $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_priv_s2_g15p.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_XSPLIT -DBC_GRID_G=15 -DBC_PORTALREP -DBC_FARSPRINT -DBC_AHIST -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@

$(OUT)/libbcvec_replay_s2_g15p.so: cpp/bc_capi_vec.cpp $(wildcard cpp/*.hpp)
	$(CXX) $(CXXFLAGS) -DBC_SONAR2 -DBC_XSPLIT -DBC_GRID_G=15 -DBC_PORTALREP -DBC_FARSPRINT -DBC_AHIST -DBC_MAX_STEPS=64 $< -o $@.tmp && mv $@.tmp $@
