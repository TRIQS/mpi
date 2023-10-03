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

#pragma once

#include <itertools/itertools.hpp>

#include <mpi.h>
#include <cassert>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <unistd.h>

/// Library namespace
namespace mpi {

  /// Check if MPI was initialized
  inline bool is_initialized() noexcept {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag;
  }

  // ------------------------------------------------------------

  /**
   * Constant bool initialized by checking for MPI runtime environment.
   * Expects either OpenMPI, MPICH, IntelMPI or Cray(MPICH) environment.
   * FORCE_MPI_INIT can be used to overwrite this manually.
   */
  static const bool has_env = []() {
    if (std::getenv("OMPI_COMM_WORLD_RANK") != nullptr or std::getenv("PMI_RANK") != nullptr or std::getenv("CRAY_MPICH_VERSION") != nullptr
        or std::getenv("FORCE_MPI_INIT") != nullptr)
      return true;
    else
      return false;
  }();

  /// Environment must be initialized in C++
  struct environment {

    // MPICH does not allow Init without argc, argv, so we do not allow default constructors
    // for portability, cf #133
    environment(int argc, char *argv[]) { // NOLINT
      if (has_env && !is_initialized()) MPI_Init(&argc, &argv);
    }
    ~environment() {
      if (has_env) MPI_Finalize();
    }
  };

  // ------------------------------------------------------------

  class shared_communicator;

  /// The communicator class
  class communicator {
    friend class shared_communicator;
    MPI_Comm _com = MPI_COMM_WORLD;

    public:
    communicator() = default;

    communicator(MPI_Comm c) : _com(c) {}

    [[nodiscard]] MPI_Comm get() const noexcept { return _com; }

    [[nodiscard]] bool is_null() const noexcept { return _com == MPI_COMM_NULL; }

    [[nodiscard]] int rank() const {
      if (has_env) {
        int num = 0;
        MPI_Comm_rank(_com, &num);
        return num;
      } else
        return 0;
    }

    [[nodiscard]] int size() const {
      if (has_env) {
        int num = 0;
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
        return {};
    }

    [[nodiscard]] shared_communicator split_shared(int split_type = MPI_COMM_TYPE_SHARED, int key = 0) const;

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

    // Default barrier is implemented as a poll barrier, which only checks each msec's if
    // all ranks reached the barrier by using a non blocking MPI_Ibarrier combined with MPI_Test.
    // The default of 1 msec reduces CPU load considerably:
    // 1 msec ~ 1% cpu load
    // 10 msec ~ 0.5% cpu load
    // 100 msec ~ 0.01% cpu load
    // For very unbalanced load that takes long times to finish, 1000msec is a good choice.
    // If poll_msec==0 the classic MPI_Barrier will be called.
    void barrier(long poll_msec = 1) {
      if (has_env) {
        if (poll_msec == 0) {
          MPI_Barrier(_com);
        } else {
          MPI_Request req{};
          int flag = 0;
          // non blocking barrier to check which rank is here
          MPI_Ibarrier(_com, &req);
          // check each poll_msec via MPI_Test if all ranks reached the barrier
          do {
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            // convert to millisec
            usleep(poll_msec * 1000);
          } while (!flag);
        }
      }
    }
  };

  /// The shared communicator class
  class shared_communicator : public communicator {};

  [[nodiscard]] shared_communicator communicator::split_shared(int split_type, int key) const {
    if (has_env) {
      shared_communicator c;
      MPI_Comm_split_type(_com, split_type, key, MPI_INFO_NULL, &c._com);
      return c;
    } else
      return {};
  }

  // ----------------------------------------
  // ------- MPI Lazy Struct and Tags -------
  // ----------------------------------------

  namespace tag {
    struct reduce {};
    struct scatter {};
    struct gather {};
  } // namespace tag

  // A small lazy tagged class
  template <typename Tag, typename T> struct lazy {
    T rhs;
    communicator c;
    int root{};
    bool all{};
    MPI_Op op{};
  };

  // ----------------------------------------
  // ------- general functions -------
  // ----------------------------------------

  /**
   * Generic broadcast implementation
   */
  template <typename T> [[gnu::always_inline]] void broadcast(T &x, communicator c = {}, int root = 0) {
    static_assert(not std::is_const_v<T>, "mpi::broadcast cannot be called on const objects");
    if (has_env) mpi_broadcast(x, c, root);
  }

  namespace detail {

    template <typename T> inline constexpr bool is_mpi_lazy = false;

    template <typename Tag, typename T> inline constexpr bool is_mpi_lazy<lazy<Tag, T>> = true;

    template <typename T> inline constexpr bool is_std_vector = false;

    template <typename T> inline constexpr bool is_std_vector<std::vector<T>> = true;

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

  template <typename T>
  [[gnu::always_inline]] inline decltype(auto) reduce(T &&x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
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

  template <typename T>
  [[gnu::always_inline]] inline void reduce_in_place(T &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    static_assert(not std::is_const_v<T>, "In-place mpi functions cannot be called on const objects");
    if (has_env) mpi_reduce_in_place(x, c, root, all, op);
  }

  template <typename T> [[gnu::always_inline]] inline decltype(auto) scatter(T &&x, mpi::communicator c = {}, int root = 0) {
    using r_t = decltype(mpi_scatter(std::forward<T>(x), c, root));

    if constexpr (detail::is_mpi_lazy<r_t>) {
      return mpi_scatter(std::forward<T>(x), c, root);
    } else {
      // if it does not have a mpi lazy type, check manually if triqs is run with MPI
      if (has_env)
        return mpi_scatter(std::forward<T>(x), c, root);
      else
        return detail::convert<r_t>(std::forward<T>(x));
    }
  }

  template <typename T> [[gnu::always_inline]] inline decltype(auto) gather(T &&x, mpi::communicator c = {}, int root = 0, bool all = false) {
    using r_t = decltype(mpi_gather(std::forward<T>(x), c, root, all));

    if constexpr (detail::is_mpi_lazy<r_t>) {
      return mpi_gather(std::forward<T>(x), c, root, all);
    } else {
      // if it does not have a mpi lazy type, check manually if triqs is run with MPI
      if (has_env)
        return mpi_gather(std::forward<T>(x), c, root, all);
      else
        return detail::convert<r_t>(std::forward<T>(x));
    }
  }

  template <typename T> [[gnu::always_inline]] inline decltype(auto) all_reduce(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    return reduce(std::forward<T>(x), c, 0, true, op);
  }

  template <typename T> [[gnu::always_inline]] inline void all_reduce_in_place(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) {
    reduce_in_place(std::forward<T>(x), c, 0, true, op);
  }

  template <typename T> [[gnu::always_inline]] inline decltype(auto) all_gather(T &&x, communicator c = {}) {
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
  template <typename T> struct mpi_type {};

#define D(T, MPI_TY)                                                                                                                                 \
  template <> struct mpi_type<T> {                                                                                                                   \
    static MPI_Datatype get() noexcept { return MPI_TY; }                                                                                            \
  }
  template <typename T> struct mpi_type<const T> : mpi_type<T> {};
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

  // Helper Functions
  template <typename T, typename = void> constexpr bool has_mpi_type                              = false;
  template <typename T> constexpr bool has_mpi_type<T, std::void_t<decltype(mpi_type<T>::get())>> = true;

  namespace detail {

    template <typename... T, size_t... Is> void _init_mpi_tuple_displ(std::index_sequence<Is...>, std::tuple<T...> _tie, MPI_Aint *disp) {
      ((void)(disp[Is] = {(char *)&std::get<Is>(_tie) - (char *)&std::get<0>(_tie)}), ...);
    }
  } // namespace detail

  template <typename... T> MPI_Datatype get_mpi_type(std::tuple<T...> _tie) {
    static constexpr int N = sizeof...(T);
    MPI_Datatype types[N]  = {mpi_type<std::remove_reference_t<T>>::get()...};

    int blocklen[N];
    for (int i = 0; i < N; ++i) { blocklen[i] = 1; }
    MPI_Aint disp[N];
    detail::_init_mpi_tuple_displ(std::index_sequence_for<T...>{}, _tie, disp);
    if (std::any_of(disp, disp + N, [](MPI_Aint i) { return i < 0; })) {
      std::cerr << "ERROR: Custom mpi types require non-negative displacements\n";
      std::abort();
    }

    MPI_Datatype cty{};
    MPI_Type_create_struct(N, blocklen, disp, types, &cty);
    MPI_Type_commit(&cty);
    return cty;
  }

  template <typename... T> struct mpi_type<std::tuple<T...>> {
    static MPI_Datatype get() noexcept { return get_mpi_type(std::tuple<T...>{}); }
  };

  // A generic implementation for a struct
  // the struct should have as_tie
  template <typename T> struct mpi_type_from_tie {
    static MPI_Datatype get() noexcept { return get_mpi_type(tie_data(T{})); }
  };

  template <class BaseType> class shared_window;

  /// The window class
  template <class BaseType>
  class window {
    friend class shared_window<BaseType>;
    MPI_Win win{MPI_WIN_NULL};
  public:
    window() = default;
    window(window const&) = delete;
    window(window &&) = delete;
    window& operator=(window const&) = delete;
    window& operator=(window &&) = delete;

    explicit window(communicator &c, BaseType *base, MPI_Aint size = 0) {
      MPI_Win_create(base, size * sizeof(BaseType), alignof(BaseType), MPI_INFO_NULL, c.get(), &win);
    }

    ~window() {
      if (win != MPI_WIN_NULL) {
        MPI_Win_free(&win);
      }
    }

    operator MPI_Win() const { return win; };
    operator MPI_Win*() { return &win; };

    void fence(int assert = 0) const {
      MPI_Win_fence(assert, win);
    }

    template <typename TargetType = BaseType, typename OriginType>
    std::enable_if_t<has_mpi_type<OriginType> && has_mpi_type<TargetType>, void>
    get(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        int target_count_ = target_count < 0 ? origin_count : target_count;
        MPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
    };

    template <typename TargetType = BaseType, typename OriginType>
    std::enable_if_t<has_mpi_type<OriginType> && has_mpi_type<TargetType>, void>
    put(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        int target_count_ = target_count < 0 ? origin_count : target_count;
        MPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
    };

    void* get_attr(int win_keyval) const {
      int flag;
      void *attribute_val;
      MPI_Win_get_attr(win, win_keyval, &attribute_val, &flag);
      assert(flag);
      return attribute_val;
    }
    BaseType* base() const { return static_cast<BaseType*>(get_attr(MPI_WIN_BASE)); }
    MPI_Aint size() const { return *static_cast<MPI_Aint*>(get_attr(MPI_WIN_SIZE)); }
    int disp_unit() const { return *static_cast<int*>(get_attr(MPI_WIN_DISP_UNIT)); }
  };

  /// The shared_window class
  template <class BaseType>
  class shared_window : public window<BaseType> {
  public:
    shared_window(shared_communicator& c, MPI_Aint size) {
      void* baseptr = nullptr;
      MPI_Win_allocate_shared(size * sizeof(BaseType), alignof(BaseType), MPI_INFO_NULL, c.get(), &baseptr, &(this->win));
    }

    std::tuple<MPI_Aint, int, void*> query(int rank = MPI_PROC_NULL) const {
      MPI_Aint size = 0;
      int disp_unit = 0;
      void *baseptr = nullptr;
      MPI_Win_shared_query(this->win, rank, &size, &disp_unit, &baseptr);
      return {size, disp_unit, baseptr};
    }

    MPI_Aint size(int rank = MPI_PROC_NULL) const { return std::get<0>(query(rank)) / sizeof(BaseType); }
    int disp_unit(int rank = MPI_PROC_NULL) const { return std::get<1>(query(rank)); }
    BaseType* base(int rank = MPI_PROC_NULL) const { return static_cast<BaseType*>(std::get<2>(query(rank))); }
  };

  /* -----------------------------------------------------------
  *   Custom mpi operator
  * ---------------------------------------------------------- */

  namespace detail {
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
    template <typename T> T _generic_add(T const &lhs, T const &rhs) { return lhs + rhs; }
  } // namespace detail
  /**
   * @tparam T  Type on which the function will operate
   * @tparam F  The C function to be mapped
   */
  template <typename T, T (*F)(T const &, T const &)> MPI_Op map_C_function() {
    MPI_Op myOp{};
    MPI_Op_create(detail::_map_function<T, F>, true, &myOp);
    return myOp;
  }

  /**
   * Map addition
   * @tparam T  Type on which the addition will operate
   */
  template <typename T> MPI_Op map_add() {
    MPI_Op myOp{};
    MPI_Op_create(detail::_map_function<T, detail::_generic_add<T>>, true, &myOp);
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
  template <typename T> auto chunk(T &&range, communicator comm = {}) {
    auto total_size           = itertools::distance(std::cbegin(range), std::cend(range));
    auto [start_idx, end_idx] = itertools::chunk_range(0, total_size, comm.size(), comm.rank());
    return itertools::slice(std::forward<T>(range), start_idx, end_idx);
  }

  /* -----------------------------------------------------------
  *  basic types
  * ---------------------------------------------------------- */

  // NOTE: We keep the naming mpi_XXX for the actual implementation functions
  // so they can be defined in other namespaces and the mpi::reduce(T,...) function
  // can find them via ADL
  template <typename T> std::enable_if_t<has_mpi_type<T>> mpi_broadcast(T &a, communicator c = {}, int root = 0) {
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
