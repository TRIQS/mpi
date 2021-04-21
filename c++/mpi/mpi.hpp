// Copyright (c) 2019 Simons Foundation
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

#pragma once

#include <itertools/itertools.hpp>

#include <mpi.h>
#include <complex>
#include <type_traits>
#include <algorithm>

namespace mpi {

  inline bool is_initialized() noexcept {
    int flag;
    MPI_Initialized(&flag);
    return flag;
  }

  // ------------------------------------------------------------

  /* helper function to check for MPI runtime environment
   * covers at the moment OpenMPI, MPICH, and intelmpi
   * as cray uses MPICH under the hood it should work as well 
   */
  static const bool has_env = []() {
    if (std::getenv("OMPI_COMM_WORLD_RANK") != nullptr or std::getenv("PMI_RANK") != nullptr)
      return true;
    else
      return false;
  }();

  /// Environment must be initialized in C++
  struct environment {

    // MPICH does not allow Init without argc, argv, so we do not allow default constructors
    // for portability, cf #133
    environment(int argc, char *argv[]) { // NOLINT
      if (!is_initialized()) MPI_Init(&argc, &argv);
    }
    ~environment() { MPI_Finalize(); }
  };

  // ------------------------------------------------------------

  /// The communicator. Todo : add more constructors.
  class communicator {
    MPI_Comm _com = MPI_COMM_WORLD;

    public:
    communicator() = default;

    communicator(MPI_Comm c) : _com(c) {}

    [[nodiscard]] MPI_Comm get() const noexcept { return _com; }

    [[nodiscard]] int rank() const {
      if (has_env) {
        int num;
        MPI_Comm_rank(_com, &num);
        return num;
      } else
        return 0;
    }

    [[nodiscard]] int size() const {
      if (has_env) {
        int num;
        MPI_Comm_size(_com, &num);
        return num;
      } else
        return 1;
    }

    [[nodiscard]] communicator split(int color, int key = 0) const {
      if (has_env) {
        communicator c;
        MPI_Comm_split(_com, color, key, &c._com);
        return c;
      } else
        //TODO split should not be done without MPI?
        return 0;
    }

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

    void barrier() const {
      if (has_env) { MPI_Barrier(_com); }
    }
  };

  // ----------------------------------------
  // ------- MPI Lazy Struct and Tags -------
  // ----------------------------------------

  namespace tag {
    struct reduce {};
    struct scatter {};
    struct gather {};
  } // namespace tag

  // A small lazy tagged class
  template <typename Tag, typename T>
  struct lazy {
    T rhs;
    communicator c;
    int root{};
    bool all{};
    MPI_Op op{};
  };

  // ----------------------------------------
  // ------- general functions -------
  // ----------------------------------------

  template <typename T>
  [[gnu::always_inline]] inline void broadcast(T &x, communicator c = {}, int root = 0) {
    static_assert(not std::is_const_v<T>, "mpi::broadcast cannot be called on const objects");
    if (has_env) mpi_broadcast(x, c, root);
  }

  namespace details {

    template <typename T>
    inline constexpr bool is_mpi_lazy = false;

    template <typename Tag, typename T>
    inline constexpr bool is_mpi_lazy<lazy<Tag, T>> = true;

    template <typename T>
    inline constexpr bool is_std_vector = false;

    template <typename T>
    inline constexpr bool is_std_vector<std::vector<T>> = true;

    template <typename T, typename V>
    T convert(V v) {
      if constexpr (is_std_vector<T>) {
        T res;
        res.reserve(v.size());
        for (auto &x : v) res.emplace_back(convert<typename T::value_type>(std::move(x)));
        return res;
      } else
        return T{std::move(v)};
    }
  } // namespace details

  template <typename T>
  [[gnu::always_inline]] inline decltype(auto) reduce(T &&x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    using r_t = decltype(mpi_reduce(std::forward<T>(x), c, root, all, op));

    if constexpr (details::is_mpi_lazy<r_t>) {
      return mpi_reduce(std::forward<T>(x), c, root, all, op);
    } else {
      if (has_env)
        return mpi_reduce(std::forward<T>(x), c, root, all, op);
      else
        return details::convert<r_t>(std::forward<T>(x));
    }
  }

  template <typename T>
  [[gnu::always_inline]] inline void reduce_in_place(T &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    static_assert(not std::is_const_v<T>, "In-place mpi functions cannot be called on const objects");
    if (has_env) mpi_reduce_in_place(x, c, root, all, op);
  }

  template <typename T>
  [[gnu::always_inline]] inline decltype(auto) scatter(T &&x, mpi::communicator c = {}, int root = 0) {
    using r_t = decltype(mpi_scatter(std::forward<T>(x), c, root));

    if constexpr (details::is_mpi_lazy<r_t>) {
      return mpi_scatter(std::forward<T>(x), c, root);
    } else {
      // if it does not have a mpi lazy type, check manually if triqs is run with MPI
      if (has_env)
        return mpi_scatter(std::forward<T>(x), c, root);
      else
        return details::convert<r_t>(std::forward<T>(x));
    }
  }

  template <typename T>
  [[gnu::always_inline]] inline decltype(auto) gather(T &&x, mpi::communicator c = {}, int root = 0, bool all = false) {
    using r_t = decltype(mpi_gather(std::forward<T>(x), c, root, all));

    if constexpr (details::is_mpi_lazy<r_t>) {
      return mpi_gather(std::forward<T>(x), c, root, all);
    } else {
      // if it does not have a mpi lazy type, check manually if triqs is run with MPI
      if (has_env)
        return mpi_gather(std::forward<T>(x), c, root, all);
      else
        return details::convert<r_t>(std::forward<T>(x));
    }
  }

  template <typename T>
  [[gnu::always_inline]] inline decltype(auto) all_reduce(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    return reduce(std::forward<T>(x), c, 0, true, op);
  }

  template <typename T>
  [[gnu::always_inline]] inline void all_reduce_in_place(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    reduce_in_place(std::forward<T>(x), c, 0, true, op);
  }

  template <typename T>
  [[gnu::always_inline]] inline decltype(auto) all_gather(T &&x, communicator c = {}) {
    return gather(std::forward<T>(x), c, 0, true);
  }

  template <typename T>
  [[gnu::always_inline]] [[deprecated("mpi_all_reduce is deprecated, please use mpi::all_reduce instead")]] inline decltype(auto)
  mpi_all_reduce(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    return reduce(std::forward<T>(x), c, 0, true, op);
  }

  template <typename T>
  [[gnu::always_inline]] [[deprecated("mpi_all_gather is deprecated, please use mpi::all_gather instead")]] inline decltype(auto)
  mpi_all_gather(T &&x, communicator c = {}) {
    return gather(std::forward<T>(x), c, 0, true);
  }

  /* -----------------------------------------------------------
  *   transformation type -> mpi types
  * ---------------------------------------------------------- */

  // Specialize this struct for any type with member function
  //   static MPI_Datatype get() noexcept
  // to provide its MPI type
  template <typename T>
  struct mpi_type {};

#define D(T, MPI_TY)                                                                                                                                 \
  template <>                                                                                                                                        \
  struct mpi_type<T> {                                                                                                                               \
    static MPI_Datatype get() noexcept { return MPI_TY; }                                                                                            \
  }
  D(bool, MPI_CXX_BOOL);
  D(char, MPI_CHAR);
  D(int, MPI_INT);
  D(long, MPI_LONG);
  D(long long, MPI_LONG_LONG);
  D(double, MPI_DOUBLE);
  D(float, MPI_FLOAT);
  D(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX);
  D(unsigned long, MPI_UNSIGNED_LONG);
  D(unsigned int, MPI_UNSIGNED);
  D(unsigned long long, MPI_UNSIGNED_LONG_LONG);
#undef D

  // Helper Functions
  template <typename T, typename = void>
  constexpr bool has_mpi_type = false;
  template <typename T>
  constexpr bool has_mpi_type<T, std::void_t<decltype(mpi_type<T>::get())>> = true;

  namespace details {

    template <typename... T, size_t... Is>
    void _init_mpi_tuple_displ(std::index_sequence<Is...>, std::tuple<T...> _tie, MPI_Aint *disp) {
      ((void)(disp[Is] = {(char *)&std::get<Is>(_tie) - (char *)&std::get<0>(_tie)}), ...);
    }
  } // namespace details

  template <typename... T>
  MPI_Datatype get_mpi_type(std::tuple<T...> _tie) {
    static constexpr int N = sizeof...(T);
    MPI_Datatype types[N]  = {mpi_type<std::remove_reference_t<T>>::get()...};

    int blocklen[N];
    for (int i = 0; i < N; ++i) { blocklen[i] = 1; }
    MPI_Aint disp[N];
    details::_init_mpi_tuple_displ(std::index_sequence_for<T...>{}, _tie, disp);
    if (std::any_of(disp, disp + N, [](MPI_Aint i) { return i < 0; })) {
      std::cerr << "ERROR: Custom mpi types require non-negative displacements\n";
      std::abort();
    }

    MPI_Datatype cty;
    MPI_Type_create_struct(N, blocklen, disp, types, &cty);
    MPI_Type_commit(&cty);
    return cty;
  }

  template <typename... T>
  struct mpi_type<std::tuple<T...>> {
    static MPI_Datatype get() noexcept { return get_mpi_type(std::tuple<T...>{}); }
  };

  // A generic implementation for a struct
  // the struct should have as_tie
  template <typename T>
  struct mpi_type_from_tie {
    static MPI_Datatype get() noexcept { return get_mpi_type(tie_data(T{})); }
  };

  /* -----------------------------------------------------------
  *   Custom mpi operator
  * ---------------------------------------------------------- */

  namespace details {
    // variable template that maps the function
    // for the meaning of +[](...) , cf
    // https://stackoverflow.com/questions/17822131/resolving-ambiguous-overload-on-function-pointer-and-stdfunction-for-a-lambda
    template <typename T, T (*F)(T const &, T const &)>
    MPI_User_function *_map_function = +[](void *in, void *inout, int *len, MPI_Datatype *) {
      auto *inT    = static_cast<T *>(in);
      auto *inoutT = static_cast<T *>(inout);
      for (int i = 0; i < *len; ++i, ++inT, ++inoutT) { *inoutT = F(*inoutT, *inT); }
    };

    // Generic addition
    template <typename T>
    T _generic_add(T const &lhs, T const &rhs) {
      return lhs + rhs;
    }
  } // namespace details
  /**
   * @tparam T  Type on which the function will operate
   * @tparam F  The C function to be mapped
   */
  template <typename T, T (*F)(T const &, T const &)>
  MPI_Op map_C_function() {
    MPI_Op myOp;
    MPI_Op_create(details::_map_function<T, F>, true, &myOp);
    return myOp;
  }

  /**
   * Map addition
   * @tparam T  Type on which the addition will operate
   */
  template <typename T>
  MPI_Op map_add() {
    MPI_Op myOp;
    MPI_Op_create(details::_map_function<T, details::_generic_add<T>>, true, &myOp);
    return myOp;
  }

  /* -----------------------------------------------------------
  *   Some helper function
  * ---------------------------------------------------------- */

  inline long chunk_length(long end, int n_nodes, int rank) {
    auto [node_begin, node_end] = itertools::chunk_range(0, end, n_nodes, rank);
    return node_end - node_begin;
  }

  /**
    * Function to chunk a range, distributing it uniformly over all MPI ranks.
    *
    * @tparam T The type of the range
    *
    * @param range The range to chunk
    * @param comm The mpi communicator
    */
  template <typename T>
  auto chunk(T &&range, communicator comm = {}) {
    auto total_size           = std::distance(std::cbegin(range), std::cend(range));
    auto [start_idx, end_idx] = itertools::chunk_range(0, total_size, comm.size(), comm.rank());
    return itertools::slice(std::forward<T>(range), start_idx, end_idx);
  }

  /* -----------------------------------------------------------
  *  basic types
  * ---------------------------------------------------------- */

  // NOTE: We keep the naming mpi_XXX for the actual implementation functions
  // so they can be defined in other namespaces and the mpi::reduce(T,...) function
  // can find them via ADL
  template <typename T>
  std::enable_if_t<has_mpi_type<T>> mpi_broadcast(T &a, communicator c = {}, int root = 0) {
    MPI_Bcast(&a, 1, mpi_type<T>::get(), root, c.get());
  }

  template <typename T>
  std::enable_if_t<has_mpi_type<T>, T> mpi_reduce(T const &a, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    T b;
    auto d = mpi_type<T>::get();
    if (!all)
      // Old mpi implementations may require a non-const void *sendbuf
      MPI_Reduce(const_cast<T *>(&a), &b, 1, d, op, root, c.get());
    else
      MPI_Allreduce(const_cast<T *>(&a), &b, 1, d, op, c.get());
    return b;
  }

  template <typename T>
  std::enable_if_t<has_mpi_type<T>> mpi_reduce_in_place(T &a, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    if (!all)
      MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : &a), &a, 1, mpi_type<T>::get(), op, root, c.get());
    else
      MPI_Allreduce(MPI_IN_PLACE, &a, 1, mpi_type<T>::get(), op, c.get());
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
