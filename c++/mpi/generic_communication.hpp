// Copyright (c) 2024 Simons Foundation
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
// Authors: Thomas Hahn, Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides generic implementations for a subset of collective MPI communications (broadcast, reduce, gather,
 * scatter).
 * @details The generic functions (mpi::broadcast, mpi::reduce, mpi::scatter, ...) call their more specialized
 * counterparts (e.g. mpi::mpi_broadcast, mpi::mpi_reduce, mpi::mpi_scatter, ...). They depend on ADL.
 */

#pragma once

#include "./datatypes.hpp"
#include "./lazy.hpp"
#include "./utils.hpp"

#include <mpi.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  namespace detail {

    // Type trait to check if a type is a std::vector.
    template <typename T> inline constexpr bool is_std_vector = false;

    // Spezialization of is_std_vector for std::vector<T>.
    template <typename T> inline constexpr bool is_std_vector<std::vector<T>> = true;

    // Convert an object of type V to an object of type T.
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
   * @brief Generic MPI broadcast.
   *
   * @details If mpi::has_env is true, this function calls the specialized `mpi_broadcast` function for the given
   * object, otherwise it does nothing.
   *
   * @tparam T Type to be broadcasted.
   * @param x Object to be broadcasted.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> [[gnu::always_inline]] void broadcast(T &&x, communicator c = {}, int root = 0) {
    static_assert(not std::is_const_v<T>, "mpi::broadcast cannot be called on const objects");
    if (has_env) mpi_broadcast(std::forward<T>(x), c, root);
  }

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
    if constexpr (is_mpi_lazy<r_t>) {
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
   * @details If mpi::has_env is true, this functions calls the specialized `mpi_reduce_in_place` function for the given
   * object. Otherwise, it does nothing.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T>
  [[gnu::always_inline]] inline void reduce_in_place(T &&x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    static_assert(not std::is_const_v<T>, "In-place mpi functions cannot be called on const objects");
    if (has_env) mpi_reduce_in_place(std::forward<T>(x), c, root, all, op);
  }

  /**
   * @brief Generic MPI scatter.
   *
   * @details If mpi::has_env is true or if the return type of the specialized `mpi_scatter` is lazy, this function
   * calls the specialized `mpi_scatter` function for the given object. Otherwise, it simply converts the input object
   * to the output type `mpi_scatter` would return.
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
    if constexpr (is_mpi_lazy<r_t>) {
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
   * calls the specialized `mpi_gather` function for the given object. Otherwise, it simply converts the input object to
   * the output type `mpi_gather` would return.
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
    if constexpr (is_mpi_lazy<r_t>) {
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
   * @brief Implementation of an MPI broadcast for types that have a corresponding MPI datatype, i.e. for which a
   * specialization of mpi::mpi_type has been defined.
   *
   * @details It throws an exception in case a call to the MPI C library fails.
   *
   * @tparam T Type to be broadcasted.
   * @param x Object to be broadcasted.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T>
    requires(has_mpi_type<T>)
  void mpi_broadcast(T &x, communicator c = {}, int root = 0) {
    check_mpi_call(MPI_Bcast(&x, 1, mpi_type<T>::get(), root, c.get()), "MPI_Bcast");
  }

  /**
   * @brief Implementation of an MPI reduce for types that have a corresponding MPI datatype, i.e. for which a
   * specialization of mpi::mpi_type has been defined.
   *
   * @details It throws an exception in case a call to the MPI C library fails.
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
    requires(has_mpi_type<T>)
  T mpi_reduce(T const &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    T b;
    auto d = mpi_type<T>::get();
    if (!all)
      // old MPI implementations may require a non-const send buffer
      check_mpi_call(MPI_Reduce(const_cast<T *>(&x), &b, 1, d, op, root, c.get()), "MPI_Reduce"); // NOLINT
    else
      check_mpi_call(MPI_Allreduce(const_cast<T *>(&x), &b, 1, d, op, c.get()), "MPI_Allreduce"); // NOLINT
    return b;
  }

  /**
   * @brief Implementation of an in-place MPI reduce for types that have a corresponding MPI datatype, i.e. for which
   * a specialization of mpi::mpi_type has been defined.
   *
   * @details It throws an exception in case a call to the MPI C library fails.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T>
    requires(has_mpi_type<T>)
  void mpi_reduce_in_place(T &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    if (!all)
      check_mpi_call(MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : &x), &x, 1, mpi_type<T>::get(), op, root, c.get()), "MPI_Reduce");
    else
      check_mpi_call(MPI_Allreduce(MPI_IN_PLACE, &x, 1, mpi_type<T>::get(), op, c.get()), "MPI_Allreduce");
  }

  /**
   * @brief Checks if a given object is equal across all ranks in the given communicator.
   *
   * @details It requires that there is a specialized `mpi_reduce` for the given type `T` and that it is equality
   * comparable as well as default constructible.
   *
   * It makes two calls to mpi::all_reduce, one with `MPI_MIN` and the other with `MPI_MAX`, and compares their results.
   *
   * @note `MPI_MIN` and `MPI_MAX` need to make sense for the given type `T`.
   *
   * @tparam T Type to be checked.
   * @param x Object to be equality compared.
   * @param c mpi::communicator.
   * @return If the given object is equal on all ranks, it returns true. Otherwise, it returns false.
   */
  template <typename T> bool all_equal(T const &x, communicator c = {}) {
    if (!has_env) return true;
    auto min_obj = all_reduce(x, c, MPI_MIN);
    auto max_obj = all_reduce(x, c, MPI_MAX);
    return min_obj == max_obj;
  }

  /** @} */

} // namespace mpi
