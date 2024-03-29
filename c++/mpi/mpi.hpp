// Copyright (c) 2019-2022 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a simple C++ wrapper for the MPI C library.
 */

#pragma once

#include <itertools/itertools.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace mpi {

  // ---------------- MPI Environment ----------------

  /**
   * @brief Check if MPI has been initialized.
   * @return True if `MPI_Init` has been called, false otherwise.
   */
  [[nodiscard]] inline bool is_initialized() noexcept {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag;
  }

  /**
   * @brief Boolean variable that is true, if one of the environment variables `OMPI_COMM_WORLD_RANK`,
   * `PMI_RANK`, `CRAY_MPICH_VERSION` or `FORCE_MPI_INIT` is set, false otherwise.
   *
   * @details The environment variables are set, when a program is executed with `mpirun` or `mpiexec`.
   */
  static const bool has_env = []() {
    if (std::getenv("OMPI_COMM_WORLD_RANK") != nullptr or std::getenv("PMI_RANK") != nullptr or std::getenv("CRAY_MPICH_VERSION") != nullptr
        or std::getenv("FORCE_MPI_INIT") != nullptr)
      return true;
    else
      return false;
  }();

  /**
   * @brief RAII class to initialize and finalize MPI.
   *
   * @details Calls `MPI_Init` upon construction and `MPI_Finalize` upon destruction i.e. when the environment object goes out of scope.
   * If mpi::has_env is false, this struct does nothing.
   */
  struct environment {
    /**
     * @brief Construct a new mpi environment object by calling `MPI_Init`.
     *
     * @details Checks first if the program is run with an MPI runtime environment and if it has not been
     * initialized before to avoid errors.
     *
     * @param argc Number of command line arguments.
     * @param argv Command line arguments.
     */
    environment(int argc, char *argv[]) { //  NOLINT (C-style array is wanted here)
      if (has_env && !is_initialized()) MPI_Init(&argc, &argv);
    }

    /**
     * @brief Destroy the mpi environment object by calling `MPI_Finalize`.
     *
     * @details Checks first if the program is run with an MPI runtime environment and if it has not been
     * finalized before to avoid errors. Called automatically when the environment object goes out of scope.
     */
    ~environment() {
      if (has_env) MPI_Finalize();
    }
  };

  // ---------------- MPI Communicator ----------------

  /**
   * @brief C++ wrapper around `MPI_Comm` providing various convenience functions.
   *
   * @details It stores an `MPI_Comm` object as its only member which by default is set to `MPI_COMM_WORLD`.
   * Note that copying the communicator simply copies the `MPI_Comm` object, without calling `MPI_Comm_dup`.
   */
  class communicator {
    /// Wrapped `MPI_Comm` object.
    MPI_Comm _com = MPI_COMM_WORLD;

    public:
    /// Construct a communicator with `MPI_COMM_WORLD`.
    communicator() = default;

    /**
     * @brief Construct a communicator with a given `MPI_Comm` object.
     * @details The `MPI_Comm` object is copied without calling `MPI_Comm_dup`.
     */
    communicator(MPI_Comm c) : _com(c) {}

    /// Get the wrapped `MPI_Comm` object.
    [[nodiscard]] MPI_Comm get() const noexcept { return _com; }

    /**
     * @brief Get the rank of the calling process in the communicator.
     * @return The result of `MPI_Comm_rank` if mpi::has_env is true, otherwise 0.
     */
    [[nodiscard]] int rank() const {
      if (has_env) {
        int num = 0;
        MPI_Comm_rank(_com, &num);
        return num;
      } else
        return 0;
    }

    /**
     * @brief Get the size of the communicator.
     * @return The result of `MPI_Comm_size` if mpi::has_env is true, otherwise 1.
     */
    [[nodiscard]] int size() const {
      if (has_env) {
        int num = 0;
        MPI_Comm_size(_com, &num);
        return num;
      } else
        return 1;
    }

    /**
     * @brief Split the communicator into disjoint subgroups.
     *
     * @details Calls `MPI_Comm_split` with the given color and key arguments. See the MPI documentation for more details,
     * e.g. <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_split.3.html">open-mpi docs</a>.
     *
     * @return If mpi::has_env is true, return the split `MPI_Comm` object wrapped in a new communicator, otherwise
     * return a default constructed communicator.
     */
    [[nodiscard]] communicator split(int color, int key = 0) const {
      if (has_env) {
        communicator c;
        MPI_Comm_split(_com, color, key, &c._com);
        return c;
      } else
        return {};
    }

    /**
     * @brief If mpi::has_env is true, `MPI_Abort` is called with the given error code, otherwise std::abort is called.
     * @param error_code The error code to pass to `MPI_Abort`.
     */
    void abort(int error_code) {
      if (has_env)
        MPI_Abort(_com, error_code);
      else
        std::abort();
    }

#ifdef BOOST_MPI_HPP
    // Conversion to and from boost communicator, Keep for backward compatibility
    inline operator boost::mpi::communicator() const { return boost::mpi::communicator(_com, boost::mpi::comm_duplicate); }
    inline communicator(boost::mpi::communicator c) : _com(c) {}
#endif

    /**
     * @brief Barrier synchronization.
     *
     * @details Does nothing if mpi::has_env is false. Otherwise, it either uses a blocking `MPI_Barrier`
     * (if the given argument is 0) or a a non-blocking `MPI_Ibarrier` call. The given parameter determines
     * in milliseconds how often each process calls `MPI_Test` to check if all processes have reached the barrier.
     * This can considerably reduce the CPU load:
     *     - 1 msec ~ 1% cpu load
     *     - 10 msec ~ 0.5% cpu load
     *     - 100 msec ~ 0.01% cpu load
     * For a very unbalanced load that takes a long time to finish, 1000 msec is a good choice.
     *
     * @param poll_msec The polling interval in milliseconds. If set to 0, a simple `MPI_Barrier` call is used.
     */
    void barrier(long poll_msec = 1) {
      if (has_env) {
        if (poll_msec == 0) {
          MPI_Barrier(_com);
        } else {
          MPI_Request req{};
          int flag = 0;
          MPI_Ibarrier(_com, &req);
          while (!flag) {
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            usleep(poll_msec * 1000);
          }
        }
      }
    }
  };

  // ---------------- MPI Lazy ----------------

  namespace tag {

    /// Tag to specify a lazy MPI reduce call.
    struct reduce {};

    /// Tag to specify a lazy MPI scatter call.
    struct scatter {};

    /// Tag to specify a lazy MPI gather call.
    struct gather {};

  } // namespace tag

  /**
   * @brief Represents a lazy MPI communication.
   *
   * @tparam Tag Type to specify the kind of MPI communication.
   * @tparam T Type to be communicated.
   */
  template <typename Tag, typename T> struct lazy {
    /// Object to be communicated.
    T rhs;

    /// mpi::communicator used in the lazy communication.
    communicator c;

    /// Rank of the root process.
    int root{};

    /// Should we use `MPI_Allxxx`?
    bool all{};

    /// `MPI_Op` used in the lazy communication (only relevant if tag::reduce is used).
    MPI_Op op{};
  };

  // ---------------- Generic communication ----------------

  /**
   * @brief Generic MPI broadcast.
   *
   * @details If mpi::has_env is true, this function calls the specialized `mpi_broadcast` function for the given object,
   * otherwise it does nothing.
   *
   * @tparam T Type to be broadcasted.
   * @param x Object to be broadcasted.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> [[gnu::always_inline]] void broadcast(T &x, communicator c = {}, int root = 0) {
    static_assert(not std::is_const_v<T>, "mpi::broadcast cannot be called on const objects");
    if (has_env) mpi_broadcast(x, c, root);
  }

  namespace detail {

    /**
     * @brief Type trait to check if a type is mpi::lazy.
     * @tparam T Type to be checked.
     */
    template <typename T> inline constexpr bool is_mpi_lazy = false;

    /**
     * @brief Spezialization of is_mpi_lazy.
     *
     * @tparam Tag Type to specify the kind of MPI call.
     * @tparam T Type to be checked.
     */
    template <typename Tag, typename T> inline constexpr bool is_mpi_lazy<lazy<Tag, T>> = true;

    /**
     * @brief Type trait to check if a type is a std::vector.
     * @tparam T Type to be checked.
     */
    template <typename T> inline constexpr bool is_std_vector = false;

    /**
     * @brief Spezialization of is_std_vector for std::vector<T>.
     * @tparam T Value type of the std::vector.
     */
    template <typename T> inline constexpr bool is_std_vector<std::vector<T>> = true;

    /**
     * @brief Convert an object of type V to an object of type T.
     *
     * @details If V is a std::vector, the function creates a new std::vector of type T and moves the elements
     * of the input vector into the new vector. Otherwise, it simply constructs a new object of type T from the input
     * object.
     *
     * @tparam T Output type.
     * @tparam V Input type.
     * @param v Input object of type V.
     * @return Converted object of type T.
     */
    template <typename T, typename V> T convert(V v) {
      if constexpr (is_std_vector<T>) {
        T res;
        res.reserve(v.size());
        for (auto &x : v) res.emplace_back(convert<typename T::value_type>(std::move(x)));
        return res;
      } else
        return T{std::move(v)};
    }

  } // namespace detail

  /**
   * @brief Generic MPI reduce.
   *
   * @details If mpi::has_env is true or if the return type of the specialized `mpi_reduce` is lazy, this function calls
   * the specialized `mpi_reduce` function for the given object. Otherwise, it simply converts the input object to the
   * output type `mpi_reduce` would return.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return The result of the specialized `mpi_reduce` call.
   */
  template <typename T>
  [[gnu::always_inline]] inline decltype(auto) reduce(T &&x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    // return type of mpi_reduce
    using r_t = decltype(mpi_reduce(std::forward<T>(x), c, root, all, op));
    if constexpr (detail::is_mpi_lazy<r_t>) {
      return mpi_reduce(std::forward<T>(x), c, root, all, op);
    } else {
      if (has_env)
        return mpi_reduce(std::forward<T>(x), c, root, all, op);
      else
        return detail::convert<r_t>(std::forward<T>(x));
    }
  }

  /**
   * @brief Generic in-place MPI reduce.
   *
   * @details If mpi::has_env is true, this functions calls the specialized `mpi_reduce_in_place` function for the given object.
   * Otherwise, it does nothing.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T>
  [[gnu::always_inline]] inline void reduce_in_place(T &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    static_assert(not std::is_const_v<T>, "In-place mpi functions cannot be called on const objects");
    if (has_env) mpi_reduce_in_place(x, c, root, all, op);
  }

  /**
   * @brief Generic MPI scatter.
   *
   * @details If mpi::has_env is true or if the return type of the specialized `mpi_scatter` is lazy, this function
   * calls the specialized `mpi_scatter` function for the given object. Otherwise, it simply converts the input
   * object to the output type `mpi_scatter` would return.
   *
   * @tparam T Type to be scattered.
   * @param x Object to be scattered.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @return The result of the specialized `mpi_scatter` call.
   */
  template <typename T> [[gnu::always_inline]] inline decltype(auto) scatter(T &&x, mpi::communicator c = {}, int root = 0) {
    // return type of mpi_scatter
    using r_t = decltype(mpi_scatter(std::forward<T>(x), c, root));
    if constexpr (detail::is_mpi_lazy<r_t>) {
      return mpi_scatter(std::forward<T>(x), c, root);
    } else {
      if (has_env)
        return mpi_scatter(std::forward<T>(x), c, root);
      else
        return detail::convert<r_t>(std::forward<T>(x));
    }
  }

  /**
   * @brief Generic MPI gather.
   *
   * @details If mpi::has_env is true or if the return type of the specialized `mpi_gather` is lazy, this function
   * calls the specialized `mpi_gather` function for the given object. Otherwise, it simply converts the input
   * object to the output type `mpi_gather` would return.
   *
   * @tparam T Type to be gathered.
   * @param x Object to be gathered.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   * @return The result of the specialized `mpi_gather` call.
   */
  template <typename T> [[gnu::always_inline]] inline decltype(auto) gather(T &&x, mpi::communicator c = {}, int root = 0, bool all = false) {
    // return type of mpi_gather
    using r_t = decltype(mpi_gather(std::forward<T>(x), c, root, all));
    if constexpr (detail::is_mpi_lazy<r_t>) {
      return mpi_gather(std::forward<T>(x), c, root, all);
    } else {
      if (has_env)
        return mpi_gather(std::forward<T>(x), c, root, all);
      else
        return detail::convert<r_t>(std::forward<T>(x));
    }
  }

  /**
   * @brief Generic MPI all-reduce.
   * @details It simply calls mpi::reduce with `all = true`.
   */
  template <typename T> [[gnu::always_inline]] inline decltype(auto) all_reduce(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    return reduce(std::forward<T>(x), c, 0, true, op);
  }

  /**
   * @brief Generic MPI all-reduce in-place.
   * @details It simply calls mpi::reduce_in_place with `all = true`.
   */
  template <typename T> [[gnu::always_inline]] inline void all_reduce_in_place(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    reduce_in_place(std::forward<T>(x), c, 0, true, op);
  }

  /**
   * @brief Generic MPI all-gather.
   * @details It simply calls mpi::gather with `all = true`.
   */
  template <typename T> [[gnu::always_inline]] inline decltype(auto) all_gather(T &&x, communicator c = {}) {
    return gather(std::forward<T>(x), c, 0, true);
  }

  /**
   * @brief Generic MPI all-reduce.
   * @deprecated Use mpi::all_reduce instead.
   */
  template <typename T>
  [[gnu::always_inline]] [[deprecated("mpi_all_reduce is deprecated, please use mpi::all_reduce instead")]] inline decltype(auto)
  mpi_all_reduce(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    return reduce(std::forward<T>(x), c, 0, true, op);
  }

  /**
   * @brief Generic MPI all-reduce.
   * @deprecated Use mpi::all_gather instead.
   */
  template <typename T>
  [[gnu::always_inline]] [[deprecated("mpi_all_gather is deprecated, please use mpi::all_gather instead")]] inline decltype(auto)
  mpi_all_gather(T &&x, communicator c = {}) {
    return gather(std::forward<T>(x), c, 0, true);
  }

  // ---------------- MPI Datatypes ----------------

  /**
   * @brief Map C++ datatypes to the corresponding MPI datatypes.
   *
   * @details C++ types which have a corresponding MPI datatype should specialize this struct. It is assumed that it
   * has a static member function `get` which returns the `MPI_Datatype` object for a given C++ type. For example:
   *
   * @code{.cpp}
   * template <> struct mpi_type<int> {
   *   static MPI_Datatype get() noexcept { return MPI_INT; }
   * }
   * @endcode
   *
   * @tparam T C++ datatype.
   */
  template <typename T> struct mpi_type {};

#define D(T, MPI_TY)                                                                                                                                 \
  /** @brief Specialization of mpi_type for T. */                                                                                                    \
  template <> struct mpi_type<T> {                                                                                                                   \
    static MPI_Datatype get() noexcept { return MPI_TY; }                                                                                            \
  }

  // mpi_type specialization for various built-in types
  D(bool, MPI_C_BOOL);
  D(char, MPI_CHAR);
  D(int, MPI_INT);
  D(long, MPI_LONG);
  D(long long, MPI_LONG_LONG);
  D(double, MPI_DOUBLE);
  D(float, MPI_FLOAT);
  D(std::complex<double>, MPI_C_DOUBLE_COMPLEX);
  D(unsigned long, MPI_UNSIGNED_LONG);
  D(unsigned int, MPI_UNSIGNED);
  D(unsigned long long, MPI_UNSIGNED_LONG_LONG);
#undef D

  /**
   * @brief Specialization of mpi_type for `const` types.
   * @tparam T C++ type.
   */
  template <typename T> struct mpi_type<const T> : mpi_type<T> {};

  /**
   * @brief Type trait to check if a type T has a corresponding MPI datatype, i.e. if mpi_type has been specialized.
   * @tparam T Type to be checked.
   */
  template <typename T, typename = void> constexpr bool has_mpi_type = false;

  /**
   * @brief Specialization of has_mpi_type for types which have a corresponding MPI datatype.
   * @tparam T Type to be checked.
   */
  template <typename T> constexpr bool has_mpi_type<T, std::void_t<decltype(mpi_type<T>::get())>> = true;

  namespace detail {

    /**
     * @brief Helper function to get the memory displacements of the different elements in a tuple w.r.t. the first element.
     *
     * @tparam Ts Tuple element types.
     * @tparam Is Indices.
     * @param tup Tuple object.
     * @param disp Pointer to an array of memory displacements.
     */
    template <typename... T, size_t... Is> void _init_mpi_tuple_displ(std::index_sequence<Is...>, std::tuple<T...> tup, MPI_Aint *disp) {
      ((void)(disp[Is] = {(char *)&std::get<Is>(tup) - (char *)&std::get<0>(tup)}), ...);
    }

  } // namespace detail

  /**
   * @brief Create a new `MPI_Datatype` from a tuple.
   *
   * @details The tuple element types must have corresponding MPI datatypes, i.e. they must have mpi_type
   * specializtions. It uses `MPI_Type_create_struct` to create a new datatype consisting of the tuple element types.
   *
   * @tparam Ts Tuple element types.
   * @param tup Tuple object.
   * @return `MPI_Datatype` consisting of the types of the tuple elements.
   */
  template <typename... Ts> MPI_Datatype get_mpi_type(std::tuple<Ts...> tup) {
    static constexpr int N            = sizeof...(Ts);
    std::array<MPI_Datatype, N> types = {mpi_type<std::remove_reference_t<Ts>>::get()...};

    // the number of elements per type (we want 1 per type)
    std::array<int, N> blocklen;
    for (int i = 0; i < N; ++i) { blocklen[i] = 1; }

    // displacements of the blocks in bytes w.r.t. to the memory address of the first block
    std::array<MPI_Aint, N> disp;
    detail::_init_mpi_tuple_displ(std::index_sequence_for<Ts...>{}, tup, disp.data());
    if (std::any_of(disp.begin(), disp.end(), [](MPI_Aint i) { return i < 0; })) {
      std::cerr << "ERROR: Custom mpi types require non-negative displacements\n";
      std::abort();
    }

    // create and return MPI datatype
    MPI_Datatype cty{};
    MPI_Type_create_struct(N, blocklen.data(), disp.data(), types.data(), &cty);
    MPI_Type_commit(&cty);
    return cty;
  }

  /**
   * @brief Specialization of mpi_type for std::tuple.
   * @tparam Ts Tuple element types.
   */
  template <typename... T> struct mpi_type<std::tuple<T...>> {
    static MPI_Datatype get() noexcept { return get_mpi_type(std::tuple<T...>{}); }
  };

  /**
   * @brief Create an `MPI_Datatype` from some struct.
   *
   * @details It is assumed that there is a free function `tie_data` which returns a tuple containing the data
   * members of the given type. The intended use is as a base class for a specialization of mpi_type:
   *
   * @code{.cpp}
   * // type to use for MPI communication
   * struct foo {
   *   double x;
   *   int y;
   * };
   *
   * // provide a tie_data function
   * auto tie_data(foo f) {
   *   return std::tie(f.x, f.y);
   * }
   *
   * // provide a specialization of mpi_type
   * template <> struct mpi::mpi_type<foo> : mpi::mpi_type_from_tie<foo> {};
   * @endcode
   *
   * @tparam T Type to be converted to an `MPI_Datatype`.
   */
  template <typename T> struct mpi_type_from_tie {
    static MPI_Datatype get() noexcept { return get_mpi_type(tie_data(T{})); }
  };

  // ---------------- MPI Operators ----------------

  namespace detail {

    /**
     * @brief Lambda that maps a binary user function to an `MPI_User_function`.
     *
     * @details The unary plus in front of the lambda is necessary to convert it to a function pointer.
     *
     * @tparam T Type on which the binary function operates.
     * @tparam F Binary function pointer to be mapped.
     */
    template <typename T, T (*F)(T const &, T const &)>
    MPI_User_function *_map_function = +[](void *in, void *inout, int *len, MPI_Datatype *) { // NOLINT (MPI_Op_create needs a non-const pointer)
      auto *inT    = static_cast<T *>(in);
      auto *inoutT = static_cast<T *>(inout);
      for (int i = 0; i < *len; ++i, ++inT, ++inoutT) { *inoutT = F(*inoutT, *inT); }
    };

    /**
     * @brief Generic addition.
     *
     * @tparam T Type used for the addition.
     * @param lhs Left hand side summand.
     * @param rhs Right hand side summand.
     * @return Sum of given arguments.
     */
    template <typename T> T _generic_add(T const &lhs, T const &rhs) { return lhs + rhs; }

  } // namespace detail

  /**
   * @brief Create a new `MPI_Op` from a given binary function by calling `MPI_Op_create`.
   *
   * @details The binary function must have the following signature `(T const&, T const&) -> T`.
   *
   * @tparam T Type on which the binary function operates.
   * @tparam F Binary function pointer to be mapped.
   * @return `MPI_Op` created from the binary function.
   */
  template <typename T, T (*F)(T const &, T const &)> MPI_Op map_C_function() {
    MPI_Op myOp{};
    MPI_Op_create(detail::_map_function<T, F>, true, &myOp);
    return myOp;
  }

  /**
   * @brief Create a new `MPI_Op` for a generic addition by calling `MPI_Op_create`.
   *
   * @details The type is required to have an overloaded `operator+(const T& lhs, const T& rhs) -> T`.
   *
   * @tparam T Type used for the addition.
   * @return `MPI_Op` for the generic addition of the given type.
   */
  template <typename T> MPI_Op map_add() {
    MPI_Op myOp{};
    MPI_Op_create(detail::_map_function<T, detail::_generic_add<T>>, true, &myOp);
    return myOp;
  }

  // ---------------- MPI chunk ----------------

  /**
   * @brief Get the length of the i^th subrange after splitting the integer range `[0, end)` evenly across n subranges.
   *
   * @param end End of the integer range `[0, end)`.
   * @param n Number of subranges.
   * @param i Index of subrange of interest.
   * @return Length of i^th subrange.
   */
  inline long chunk_length(long end, int n, int i) {
    auto [node_begin, node_end] = itertools::chunk_range(0, end, n, i);
    return node_end - node_begin;
  }

  /**
   * @brief Divide a given range as evenly as possible across the MPI processes in a communicator and get the subrange
   * assigned to the calling process.
   *
   * @tparam R Range type.
   * @param rg Range to divide.
   * @param c mpi::communicator.
   * @return An itertools::sliced range assigned to the calling process.
   */
  template <typename R> auto chunk(R &&rg, communicator c = {}) {
    auto total_size           = itertools::distance(std::cbegin(rg), std::cend(rg));
    auto [start_idx, end_idx] = itertools::chunk_range(0, total_size, c.size(), c.rank());
    return itertools::slice(std::forward<R>(rg), start_idx, end_idx);
  }

  // ---------------- Specialized communications for MPI datatypes  ----------------

  /**
   * @brief Implementation of an MPI broadcast for types that have a corresponding MPI datatype, i.e. for which
   * a specialization of mpi_type has been defined.
   *
   * @tparam T Type to be broadcasted.
   * @param x Object to be broadcasted.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> std::enable_if_t<has_mpi_type<T>> mpi_broadcast(T &x, communicator c = {}, int root = 0) {
    MPI_Bcast(&x, 1, mpi_type<T>::get(), root, c.get());
  }

  /**
   * @brief Implementation of an MPI reduce for types that have a corresponding MPI datatype, i.e. for which
   * a specialization of mpi_type has been defined.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return The result of the reduction.
   */
  template <typename T>
  std::enable_if_t<has_mpi_type<T>, T> mpi_reduce(T const &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    T b;
    auto d = mpi_type<T>::get();
    if (!all)
      // old MPI implementations may require a non-const send buffer
      MPI_Reduce(const_cast<T *>(&x), &b, 1, d, op, root, c.get()); // NOLINT (or should we remove the const_cast?)
    else
      MPI_Allreduce(const_cast<T *>(&x), &b, 1, d, op, c.get()); // NOLINT (or should we remove the const_cast?)
    return b;
  }

  /**
   * @brief Implementation of an in-place MPI reduce for types that have a corresponding MPI datatype, i.e. for which
   * a specialization of mpi_type has been defined.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T>
  std::enable_if_t<has_mpi_type<T>> mpi_reduce_in_place(T &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    if (!all)
      MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : &x), &x, 1, mpi_type<T>::get(), op, root, c.get());
    else
      MPI_Allreduce(MPI_IN_PLACE, &x, 1, mpi_type<T>::get(), op, c.get());
  }

#define MPI_TEST_MAIN                                                                                                                                \
  int main(int argc, char **argv) {                                                                                                                  \
    ::testing::InitGoogleTest(&argc, argv);                                                                                                          \
    if (mpi::has_env) {                                                                                                                              \
      mpi::environment env(argc, argv);                                                                                                              \
      std::cout << "MPI environment detected\n";                                                                                                     \
      return RUN_ALL_TESTS();                                                                                                                        \
    } else                                                                                                                                           \
      return RUN_ALL_TESTS();                                                                                                                        \
  }

} // namespace mpi
