// zig's wasm32-wasi libc++ is built without exception support. helper.hpp only
// throws on malformed input, which never happens when metering real play, so
// a throw here just traps.
extern "C" void* __cxa_allocate_exception(__SIZE_TYPE__) { __builtin_trap(); }
extern "C" void __cxa_throw(void*, void*, void (*)(void*)) { __builtin_trap(); }
