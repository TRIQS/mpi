@page sanitizers Sanitizer builds and Open MPI 5+

[TOC]

LLVM sanitizers (ASan / UBSan / TSan / MSan) are incompatible with Open MPI
5+'s default UCX transport. **mpi** detects this combination and automatically
works around it.

@section sanitizers_symptom Symptom

With a sanitizer-enabled build linked against Open MPI 5+, every MPI process
crashes early in `MPI_Init` with a SEGV inside `libucs.so`
(`ucs_handle_error` / `ucs_debug_print_backtrace` near the top of the stack).
The same code on Open MPI 4.x runs cleanly.

@section sanitizers_cause Cause

Open MPI 5+ uses the `pml/ucx` layer by default. UCX's `libucm` installs
malloc/mmap **relocation hooks** to track RDMA buffers; LLVM sanitizers
install their own interceptors on the same symbols. The double-patched malloc
table desynchronizes ASan's shadow map, and the next allocation routed
through the wrong path lands on an unmapped page.

This is a known upstream issue with no fix at the time of writing:

  - [open-mpi/ompi#13069](https://github.com/open-mpi/ompi/issues/13069) —
    Segmentation fault when using address sanitizer (Open MPI 5.0.6, clang
    and gcc, both crash around `MPI_Init`).
  - [openucx/ucx#5030](https://github.com/openucx/ucx/issues/5030) — upstream
    UCX issue tracing the crash to `ucm/malloc_hook.cc` under ASan.

@section sanitizers_fix Automatic fix

When `c++/mpi/CMakeLists.txt` sees Open MPI 5+ **and** any of
`ASAN`/`UBSAN`/`MSAN`/`TSAN` set in the cache, it appends

```cmake
--mca pml ob1 --mca btl self,vader --mca osc ^ucx
```

to the `MPIEXEC_PREFLAGS` cache variable. These flags route Open MPI through
the `ob1` PML and shared-memory BTLs, so `libucm` is never loaded. Because
every TRIQS-ecosystem `add_test` already threads `${MPIEXEC_PREFLAGS}` to
`mpiexec`, the workaround reaches **mpi**, **nda**, **triqs**, and any
downstream project with no per-project change.

Confirm it's active by looking for this configure-time line:

```text
-- Sanitizer + Open MPI 5.x detected: appending '--mca pml ob1 --mca btl self,vader --mca osc ^ucx' to MPIEXEC_PREFLAGS to avoid UCX/sanitizer SEGV
```

@section sanitizers_manual Manual workaround

If you launch MPI by hand (raw `mpiexec`, Slurm scripts, CI without `ctest`),
pass the same flags:

```console
$ mpiexec --mca pml ob1 --mca btl self,vader --mca osc ^ucx -n 2 ./my_test
```

@section sanitizers_alternatives Alternatives

  - **Preferred long-term:** rebuild Open MPI without UCX
    (`configure --without-ucx`) in your sanitizer image.
  - **Use Open MPI 4.x** for sanitizer CI — it defaults to `ob1`/`vader` and
    is unaffected.
  - **Do not** suppress the crash with `ASAN_OPTIONS=handle_segv=0`; that
    hides the corrupted shadow map and may mask real bugs.

The detection is gated on `MPI_CXX_LIBRARY_VERSION_STRING` matching
`Open MPI v<N>`, so Intel MPI, MPICH, and other implementations are
unaffected.
