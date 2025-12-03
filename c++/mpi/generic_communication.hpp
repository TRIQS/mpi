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

#include "./communicator.hpp"
#include "./datatypes.hpp"
#include "./macros.hpp"
#include "./utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <vector>

namespace mpi {

  /**
   * @ingroup utilities
   * @brief A concept that checks if a range type is contiguous and sized and has an MPI compatible value type.
   * @tparam R Range type.
   */
  template <typename R>
  concept MPICompatibleRange = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> && has_mpi_type<std::ranges::range_value_t<R>>;

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Generic MPI broadcast.
   *
   * @details It calls the specialized `mpi_broadcast` function.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T Type to be broadcasted.
   * @param x Object to be broadcasted (into).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> [[gnu::always_inline]] void broadcast(T &&x, communicator c = {}, int root = 0) { // NOLINT (forwarding is not needed)
    mpi_broadcast(x, c, root);
  }

  /**
   * @brief Generic MPI reduce.
   *
   * @details If there is a specialized `mpi_reduce` for the given type, we call it. Otherwise, we call mpi::reduce_into
   * with the given input object and a default constructed output object of type `T`.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return Result of the specialized `mpi_reduce` call.
   */
  template <typename T>
  [[gnu::always_inline]] decltype(auto) reduce(T &&x, communicator c = {}, int root = 0, bool all = false, // NOLINT (forwarding is not needed)
                                               MPI_Op op = MPI_SUM) {
    if constexpr (requires { mpi_reduce(x, c, root, all, op); }) {
      return mpi_reduce(x, c, root, all, op);
    } else {
      std::remove_cvref_t<T> res;
      reduce_into(x, res, c, root, all, op);
      return res;
    }
  }

  /**
   * @brief Generic in place MPI reduce.
   *
   * @details We call mpi::reduce_into with the given object as the input and output argument.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced (into).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T>
  [[gnu::always_inline]] void reduce_in_place(T &&x, communicator c = {}, int root = 0, bool all = false, // NOLINT (forwarding is not needed)
                                              MPI_Op op = MPI_SUM) {
    mpi_reduce_into(x, x, c, root, all, op);
  }

  /**
   * @brief Generic MPI reduce that reduces directly into an existing output object.
   *
   * @details It calls the specialized `mpi_reduce_into` function.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T1 Type to be reduced.
   * @tparam T2 Type to be reduced into.
   * @param x_in Object to be reduced.
   * @param x_out Object to be reduced into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T1, typename T2>
  [[gnu::always_inline]] void reduce_into(T1 &&x_in, T2 &&x_out, communicator c = {}, int root = 0, // NOLINT (forwarding is not needed)
                                          bool all = false, MPI_Op op = MPI_SUM) {
    mpi_reduce_into(x_in, x_out, c, root, all, op);
  }

  /**
   * @brief Generic MPI scatter.
   *
   * @details If there is a specialized `mpi_scatter` for the given type, we call it. Otherwise, we call
   * mpi::scatter_into with the given input object and a default constructed output object of type `T`.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T Type to be scattered.
   * @param x Object to be scattered.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @return Result of the specialized `mpi_scatter` call.
   */
  template <typename T>
  [[gnu::always_inline]] decltype(auto) scatter(T &&x, mpi::communicator c = {}, int root = 0) { // NOLINT (forwarding is not needed)
    if constexpr (requires { mpi_scatter(x, c, root); }) {
      return mpi_scatter(x, c, root);
    } else {
      std::remove_cvref_t<T> res;
      scatter_into(x, res, c, root);
      return res;
    }
  }

  /**
   * @brief Generic MPI scatter that scatters directly into an existing output object.
   *
   * @details It calls the specialized `mpi_scatter_into` function.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T1 Type to be scattered.
   * @tparam T2 Type to be scattered into.
   * @param x_in Object to be scattered.
   * @param x_out Object to be scattered into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T1, typename T2>
  [[gnu::always_inline]] void scatter_into(T1 &&x_in, T2 &&x_out, communicator c = {}, int root = 0) { // NOLINT (forwarding is not needed)
    mpi_scatter_into(x_in, x_out, c, root);
  }

  /**
   * @brief Generic MPI gather.
   *
   * @details If there is a specialized `mpi_gather` for the given type, we call it. Otherwise, we call mpi::gather_into
   * with the given input object and a default constructed output object of type `T`.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T Type to be gathered.
   * @param x Object to be gathered.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   * @return Result of the specialized `mpi_gather` call.
   */
  template <typename T>
  [[gnu::always_inline]] decltype(auto) gather(T &&x, communicator c = {}, int root = 0, bool all = false) { // NOLINT (forwarding is not needed)
    if constexpr (requires { mpi_gather(x, c, root, all); }) {
      return mpi_gather(x, c, root, all);
    } else {
      std::remove_cvref_t<T> res;
      gather_into(x, res, c, root, all);
      return res;
    }
  }

  /**
   * @brief Generic MPI gather that gathers directly into an existing output object.
   *
   * @details It calls the specialized `mpi_gather_into` function.
   *
   * @note We do not check if an MPI runtime environment is being used, i.e. if mpi::has_env is true. It is the
   * responsibility of the specializations to do this check, in case they make direct calls to the MPI C library.
   *
   * @tparam T1 Type to be gathered.
   * @tparam T2 Type to be gathered into.
   * @param x_in Object to be gathered.
   * @param x_out Object to be gathered into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   */
  template <typename T1, typename T2>
  [[gnu::always_inline]] void gather_into(T1 &&x_in, T2 &&x_out, communicator c = {}, int root = 0, // NOLINT (forwarding is not needed)
                                          bool all = false) {
    mpi_gather_into(x_in, x_out, c, root, all);
  }

  /**
   * @brief Generic MPI all-reduce.
   * @details It simply calls mpi::reduce with `all = true`.
   */
  template <typename T>
  [[gnu::always_inline]] decltype(auto) all_reduce(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) { // NOLINT (forwarding is not needed)
    return reduce(x, c, 0, true, op);
  }

  /**
   * @brief Generic MPI all-reduce in place.
   * @details It simply calls mpi::reduce_in_place with `all = true`.
   */
  template <typename T>
  [[gnu::always_inline]] void all_reduce_in_place(T &&x, communicator c = {}, MPI_Op op = MPI_SUM) { // NOLINT (forwarding is not needed)
    reduce_in_place(x, c, 0, true, op);
  }

  /**
   * @brief Generic MPI all-reduce that reduces directly into an existing output object.
   * @details It simply calls mpi::reduce_into with `all = true`.
   */
  template <typename T1, typename T2>
  [[gnu::always_inline]] void all_reduce_into(T1 &&x_in, T2 &&x_out, communicator c = {}, MPI_Op op = MPI_SUM) { // NOLINT (forwarding is not needed)
    return reduce_into(x_in, x_out, c, 0, true, op);
  }

  /**
   * @brief Generic MPI all-gather.
   * @details It simply calls mpi::gather with `all = true`.
   */
  template <typename T> [[gnu::always_inline]] decltype(auto) all_gather(T &&x, communicator c = {}) { // NOLINT (forwarding is not needed)
    return gather(x, c, 0, true);
  }

  /**
   * @brief Generic MPI all-gather that gathers directly into an existing output object.
   * @details It simply calls mpi::gather_into with `all = true`.
   */
  template <typename T1, typename T2>
  [[gnu::always_inline]] void all_gather_into(T1 &&x_in, T2 &&x_out, communicator c = {}) { // NOLINT (forwarding is not needed)
    return gather_into(x_in, x_out, c, 0, true);
  }

  /**
   * @brief Checks if a given object is equal across all ranks in the given communicator.
   *
   * @details It makes two calls to mpi::all_reduce, one with `MPI_MIN` and the other with `MPI_MAX`, and compares their
   * results.
   *
   * @note `MPI_MIN` and `MPI_MAX` need to make sense for the given type `T`.
   *
   * @tparam T Type to be checked.
   * @param x Object to be equality compared.
   * @param c mpi::communicator.
   * @return If the given object is equal on all ranks, it returns true. Otherwise, it returns false.
   */
  template <typename T> bool all_equal(T const &x, communicator c = {}) {
    if (!has_env || c.size() < 2) return true;
    auto min_obj = all_reduce(x, c, MPI_MIN);
    auto max_obj = all_reduce(x, c, MPI_MAX);
    return min_obj == max_obj;
  }

  /**
   * @brief Implementation of an MPI broadcast for types that have a corresponding MPI datatype.
   *
   * @details If mpi::has_env is false or if the communicator size is < 2, it does nothing. Otherwise, it calls
   * `MPI_Bcast`.
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call.
   *
   * @tparam T Type to be broadcasted.
   * @param x Object to be broadcasted (into).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T>
    requires(has_mpi_type<T>)
  void mpi_broadcast(T &x, communicator c = {}, int root = 0) {
    // in case there is no active MPI environment or if the communicator size is < 2, do nothing
    if (!has_env || c.size() < 2) return;

    // make the MPI C library call
    check_mpi_call(MPI_Bcast(&x, 1, mpi_type<T>::get(), root, c.get()), "MPI_Bcast");
  }

  /**
   * @brief Implementation of an MPI reduce for types that have a corresponding MPI datatype.
   *
   * @details If mpi::has_env is false or if the communicator size is < 2, it returns a copy of the input object.
   * Otherwise, it calls `MPI_Allreduce` or `MPI_Reduce` with a default constructed output object.
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call.
   *
   * @tparam T Type to be reduced.
   * @param x Object to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return Result of the reduction.
   */
  template <typename T>
    requires(has_mpi_type<T>)
  T mpi_reduce(T const &x, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    // in case there is no active MPI environment or if the communicator size is < 2, return the input object
    if (!has_env || c.size() < 2) return x;

    // make the MPI C library call with a default constructed output object
    T res;
    if (all) {
      check_mpi_call(MPI_Allreduce(&x, &res, 1, mpi_type<T>::get(), op, c.get()), "MPI_Allreduce");
    } else {
      check_mpi_call(MPI_Reduce(&x, &res, 1, mpi_type<T>::get(), op, root, c.get()), "MPI_Reduce");
    }
    return res;
  }

  /**
   * @brief Implementation of an MPI reduce that reduces directly into an existing output object for types that have a
   * corresponding MPI datatype.
   *
   * @details If the addresses of the input and output objects are equal, the reduction is done in place.
   *
   * If mpi::has_env is false or if the communicator size is < 2, it either does nothing (in place) or copies the input
   * into the output object. Otherwise, it calls `MPI_Allreduce` or `MPI_Reduce` (with `MPI_IN_PLACE`).
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call and it is expected that either all or 
   * none of the receiving processes choose the in place option.
   *
   * @tparam T Type to be reduced.
   * @param x_in Object to be reduced.
   * @param x_out Object to be reduced into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T>
    requires(has_mpi_type<T>)
  void mpi_reduce_into(T const &x_in, T &x_out, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    // check if the reduction is in place
    auto in_ptr         = static_cast<void const *>(&x_in);
    auto out_ptr        = static_cast<void *>(&x_out);
    bool const in_place = (in_ptr == out_ptr);
    if (all) {
      EXPECTS_WITH_MESSAGE(all_equal(static_cast<int>(in_place), c),
                           "Either zero or all receiving processes have to choose the in place option in mpi_reduce_into");
    }

    // in case there is no active MPI environment or if the communicator size is < 2, do nothing (in place) or copy
    if (!has_env || c.size() < 2) {
      if (!in_place) x_out = x_in;
      return;
    }

    // make the MPI C library call
    if (in_place && (c.rank() == root || all)) in_ptr = MPI_IN_PLACE;
    if (all) {
      check_mpi_call(MPI_Allreduce(in_ptr, out_ptr, 1, mpi_type<T>::get(), op, c.get()), "MPI_Allreduce");
    } else {
      check_mpi_call(MPI_Reduce(in_ptr, out_ptr, 1, mpi_type<T>::get(), op, root, c.get()), "MPI_Reduce");
    }
  }

  /**
   * @brief Implementation of an MPI gather for types that have a corresponding MPI datatype.
   *
   * @details It constructs an output vector, resizes it on receiving ranks to the size of the communicator and calls
   * mpi::mpi_gather_into. On non-receiving ranks the output vector is empty.
   *
   * @tparam T Type to be gathered.
   * @param x Object to be gathered.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   * @return `std::vector` containing the gathered objects.
   */
  template <typename T>
    requires(has_mpi_type<T>)
  std::vector<T> mpi_gather(T const &x, communicator c = {}, int root = 0, bool all = false) {
    std::vector<T> res(c.rank() == root || all ? c.size() : 0);
    mpi_gather_into(x, res, c, root, all);
    return res;
  }

  /**
   * @brief Implementation of an MPI gather that gathers directly into an existing output range for types that have a
   * corresponding MPI datatype.
   *
   * @details If mpi::has_env is false or if the communicator size is < 2, it copies the input object into the range.
   * Otherwise, it calls `MPI_Allgather` or `MPI_Gather`.
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call and it expects that the range size on 
   * receiving processes is equal the communicator size.
   *
   * @tparam T Type to be gathered.
   * @tparam R MPICompatibleRange type to be gathered into.
   * @param x Object to be gathered.
   * @param rg Range to be gathered into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   */
  template <typename T, MPICompatibleRange R>
    requires(has_mpi_type<T> && std::same_as<T, std::remove_cvref_t<std::ranges::range_value_t<R>>>)
  void mpi_gather_into(T const &x, R &&rg, communicator c = {}, int root = 0, bool all = false) { // NOLINT (ranges need not be forwarded)
    // check the size of the output range
    if (c.rank() == root || all) {
      EXPECTS_WITH_MESSAGE(c.size() == std::ranges::size(rg), "Output range size is not equal the number of ranks in mpi_gather_into");
    }

    // in case there is no active MPI environment or if the communicator size is < 2, copy the input into the range
    if (!has_env || c.size() < 2) {
      std::ranges::copy(std::views::single(x), std::ranges::begin(rg));
      return;
    }

    // make the MPI C library call
    using value_t = std::ranges::range_value_t<R>;
    if (all) {
      check_mpi_call(MPI_Allgather(&x, 1, mpi_type<T>::get(), std::ranges::data(rg), 1, mpi_type<value_t>::get(), c.get()), "MPI_Allgather");
    } else {
      check_mpi_call(MPI_Gather(&x, 1, mpi_type<T>::get(), std::ranges::data(rg), 1, mpi_type<value_t>::get(), root, c.get()), "MPI_Gather");
    }
  }

  /** @} */

} // namespace mpi
