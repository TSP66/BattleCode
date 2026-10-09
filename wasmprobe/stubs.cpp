// zig's wasm32-wasi libc++ is built without exception support. helper.hpp only
// throws on malformed input, which never happens when metering real play, so
// a throw here just traps.
extern "C" void* __cxa_allocate_exception(__SIZE_TYPE__) { __builtin_trap(); }
extern "C" void __cxa_throw(void*, void*, void (*)(void*)) { __builtin_trap(); }
// bc_memory.hpp (the simulator's own DragonMemory, copied into lstmbot) keeps `static thread_local`
// scratch vectors; zig's single-threaded wasi libc has no __cxa_thread_atexit to register their
// destructors with. Judge clang links fine without this. Skipping the destructors at exit is harmless.
extern "C" int __cxa_thread_atexit(void (*)(void*), void*, void*) { return 0; }
