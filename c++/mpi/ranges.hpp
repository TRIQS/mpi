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
 * @brief Provides an MPI broadcast, reduce, scatter and gather for contiguous ranges.
 */

#pragma once

#include "./chunk.hpp"
#include "./communicator.hpp"
#include "./datatypes.hpp"
#include "./environment.hpp"
#include "./generic_communication.hpp"
#include "./macros.hpp"
#include "./utils.hpp"

#include <itertools/itertools.hpp>
#include <mpi.h>

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace mpi {

  /**
   * @brief A concept that checks if a range type is contiguous and sized.
   * @tparam R Range type.
   */
  template <typename R>
  concept contiguous_sized_range = std::ranges::contiguous_range<R> && std::ranges::sized_range<R>;

  /**
   * @brief Implementation of an MPI broadcast for an mpi::contiguous_sized_range object.
   *
   * @details If mpi::has_mpi_type is true for the value type of the range, then the range is broadcasted using a simple
   * `MPI_Bcast`. Otherwise, the generic mpi::broadcast is called for each element of the range.
   *
   * It throws an exception in case a call to the MPI C library fails and it expects that the sizes of the ranges are
   * equal across all processes.
   *
   * If the ranges are empty or if mpi::has_env is false or if the communicator size is < 2, it does nothing.
   *
   * @note It is recommended to use the generic mpi::broadcast for supported types, e.g. `std::vector`, `std::array` or
   * `std::string`. It is the user's responsibility to ensure that ranges have the correct sizes.
   *
   * @code{.cpp}
   * // create a vector on all ranks
   * auto vec = std::vector<int>(5);
   *
   * if (comm.rank() == 0) {
   *   // on rank 0, initialize the vector and broadcast the first 3 elements
   *   vec = {1, 2, 3, 0, 0};
   *   mpi::broadcast_range(std::span{vec.data(), 3}, comm);
   * } else {
   *   // on other ranks, broadcast to the last 3 elements of the vector
   *   mpi::broadcast_range(std::span{vec.data() + 2, 3}, comm);
   * }
   *
   * // output result
   * for (auto x : vec) std::cout << x << " ";
   * std::cout << std::endl;
   * @endcode
   *
   * Output (with 4 processes):
   *
   * ```
   * 1 2 3 0 0
   * 0 0 1 2 3
   * 0 0 1 2 3
   * 0 0 1 2 3
   * ```
   *
   * @tparam R mpi::contiguous_sized_range type.
   * @param rg Range to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <contiguous_sized_range R> void broadcast_range(R &&rg, communicator c = {}, int root = 0) {
    // check the sizes of all ranges
    using value_t   = std::ranges::range_value_t<R>;
    auto const size = std::ranges::size(rg);
    EXPECTS_WITH_MESSAGE(all_equal(size, c), "Range sizes are not equal across all processes in mpi::broadcast_range");

    // do nothing if the range is empty, if MPI is not initialized or if the communicator size is < 2
    if (size == 0 || !has_env || c.size() < 2) return;

    // broadcast the range
    if constexpr (has_mpi_type<value_t>)
      // make an MPI C library call for MPI compatible value types
      check_mpi_call(MPI_Bcast(std::ranges::data(rg), size, mpi_type<value_t>::get(), root, c.get()), "MPI_Bcast");
    else
      // otherwise call the specialized mpi_broadcast for each element
      for (auto &val : rg) broadcast(val, c, root);
  }

  /**
   * @brief Implementation of an in-place MPI reduce for an mpi::contiguous_sized_range object.
   *
   * @details If mpi::has_mpi_type is true for the value type of the range, then the range is reduced using a simple
   * `MPI_Reduce` or `MPI_Allreduce` with `MPI_IN_PLACE`. Otherwise, the specialized `mpi_reduce_in_place` is called
   * for each element in the range.
   *
   * It throws an exception in case a call to the MPI C library fails and it expects that the sizes of the ranges are
   * equal across all processes.
   *
   * If the ranges are empty or if mpi::has_env is false or if the communicator size is < 2, it does nothing.
   *
   * @note It is recommended to use the generic mpi::reduce_in_place and mpi::all_reduce_in_place for supported types,
   * e.g. `std::vector` or `std::array`. It is the user's responsibility to ensure that ranges have the correct sizes.
   *
   * @code{.cpp}
   * // create a vector on all ranks
   * auto vec = std::vector<int>{0, 1, 2, 3, 4};
   *
   * // in-place reduce the middle elements only on rank 0
   * mpi::reduce_in_place_range(std::span{vec.data() + 1, 3}, comm);
   *
   * // output result
   * for (auto x : vec) std::cout << x << " ";
   * std::cout << std::endl;
   * @endcode
   *
   * Output (with 4 processes):
   *
   * ```
   * 0 1 2 3 4
   * 0 1 2 3 4
   * 0 1 2 3 4
   * 0 4 8 12 4
   * ```
   *
   * @tparam R mpi::contiguous_sized_range type.
   * @param rg Range to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <contiguous_sized_range R> void reduce_in_place_range(R &&rg, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    // check the sizes of all ranges
    using value_t   = std::ranges::range_value_t<R>;
    auto const size = std::ranges::size(rg);
    EXPECTS_WITH_MESSAGE(all_equal(size, c), "Range sizes are not equal across all processes in mpi::reduce_in_place_range");

    // do nothing if the range is empty, if MPI is not initialized or if the communicator size is < 2
    if (size == 0 || !has_env || c.size() < 2) return;

    // reduce the ranges
    if constexpr (has_mpi_type<value_t>) {
      // make an MPI C library call for MPI compatible value types
      auto data = std::ranges::data(rg);
      if (!all)
        check_mpi_call(MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : data), data, size, mpi_type<value_t>::get(), op, root, c.get()), "MPI_Reduce");
      else
        check_mpi_call(MPI_Allreduce(MPI_IN_PLACE, data, size, mpi_type<value_t>::get(), op, c.get()), "MPI_Allreduce");
    } else {
      // otherwise call the specialized mpi_reduce_in_place for each element
      for (auto &val : rg) mpi_reduce_in_place(val, c, root, all, op);
    }
  }

  /**
   * @brief Implementation of an MPI reduce for an mpi::contiguous_sized_range.
   *
   * @details If mpi::has_mpi_type is true for the value type of the range, then the range is reduced using a simple
   * `MPI_Reduce` or `MPI_Allreduce`. Otherwise, the specialized `mpi_reduce` is called for each element in the range.
   *
   * It throws an exception in case a call to the MPI C library fails and it expects that the sizes of the input ranges
   * are equal across all processes and that they are equal to the size of the output range on receiving processes.
   *
   * If the input ranges are empty, it does nothing. If mpi::has_env is false or if the communicator size is < 2, it
   * simply copies the input range to the output range.
   *
   * @note It is recommended to use the generic mpi::reduce and mpi::all_reduce for supported types, e.g. `std::vector`
   * or `std::array`. It is the user's responsibility to ensure that ranges have the correct sizes.
   *
   * @code{.cpp}
   * // create input and output vectors on all ranks
   * auto in_vec = std::vector<int>{0, 1, 2, 3, 4};
   * auto out_vec = std::vector<int>(in_vec.size(), 0);
   *
   * // allreduce the middle elements of the input vector to the last elements of the output vector
   * mpi::reduce_range(std::span{in_vec.data() + 1, 3}, std::span{out_vec.data() + 2, 3}, comm, 0, true);
   *
   * // output result
   * for (auto x : out_vec) std::cout << x << " ";
   * std::cout << std::endl;
   * @endcode
   *
   * Output (with 4 processes):
   *
   * ```
   * 0 0 4 8 12
   * 0 0 4 8 12
   * 0 0 4 8 12
   * 0 0 4 8 12
   * ```
   *
   * @tparam R1 mpi::contiguous_sized_range type.
   * @tparam R2 mpi::contiguous_sized_range type.
   * @param in_rg Range to reduce.
   * @param out_rg Range to reduce into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <contiguous_sized_range R1, contiguous_sized_range R2>
  void reduce_range(R1 &&in_rg, R2 &&out_rg, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    // check input and output ranges
    auto const in_size = std::ranges::size(in_rg);
    EXPECTS_WITH_MESSAGE(all_equal(in_size, c), "Input range sizes are not equal across all processes in mpi::reduce_range");
    if (c.rank() == root || all) {
      EXPECTS_WITH_MESSAGE(in_size == std::ranges::size(out_rg), "Input and output range sizes are not equal in mpi::reduce_range");
    }

    // do nothing if the input range is empty
    if (in_size == 0) return;

    // simply copy if there is no active MPI environment or if the communicator size is < 2
    if (!has_env || c.size() < 2) {
      std::ranges::copy(std::forward<R1>(in_rg), std::ranges::data(out_rg));
      return;
    }

    // reduce the ranges
    using in_value_t  = std::ranges::range_value_t<R1>;
    using out_value_t = std::ranges::range_value_t<R2>;
    if constexpr (has_mpi_type<in_value_t> && std::same_as<in_value_t, out_value_t>) {
      // make an MPI C library call for MPI compatible value types
      auto const in_data = std::ranges::data(in_rg);
      auto out_data      = std::ranges::data(out_rg);
      if (!all)
        check_mpi_call(MPI_Reduce(in_data, out_data, in_size, mpi_type<in_value_t>::get(), op, root, c.get()), "MPI_Reduce");
      else
        check_mpi_call(MPI_Allreduce(in_data, out_data, in_size, mpi_type<in_value_t>::get(), op, c.get()), "MPI_Allreduce");
    } else {
      // otherwise call the specialized mpi_reduce for each element
      // the size of the output range is arbitrary on non-recieving ranks, so we cannot use transform on them
      if (c.rank() == root || all)
        std::ranges::transform(std::forward<R1>(in_rg), std::ranges::data(out_rg), [&](auto const &val) { return reduce(val, c, root, all, op); });
      else
        // the assignment is needed in case a lazy object is returned
        std::ranges::for_each(std::forward<R1>(in_rg), [&](auto const &val) { [[maybe_unused]] out_value_t ignore = reduce(val, c, root, all, op); });
    }
  }

  /**
   * @brief Implementation of an MPI scatter for an mpi::contiguous_sized_range.
   *
   * @details If mpi::has_mpi_type is true for the value type of the range, then the range is scattered as evenly as
   * possible across the processes in the communicator using a simple `MPI_Scatterv`. Otherwise an exception is thrown.
   *
   * The user can specify a chunk size which is used to divide the input range into chunks of the specified size. The
   * number of chunks are then distributed evenly across the processes in the communicator. The size of the input range
   * is required to be a multiple of the given chunk size, otherwise an exception is thrown.
   *
   * It throws an exception in case a call to the MPI C library fails and it expects that the output ranges have the
   * correct size and that they add up to the size of the input range on the root process.
   *
   * If the input range is empty on root, it does nothing. If mpi::has_env is false or if the communicator size is < 2,
   * it simply copies the input range to the output range.
   *
   * @note It is recommended to use the generic mpi::scatter for supported types, e.g. `std::vector`. It is the user's
   * responsibility to ensure that the ranges have the correct sizes (mpi::chunk_length can be useful to do that).
   *
   * @code{.cpp}
   * // create input and output vectors on all ranks
   * auto in_vec = std::vector<int>{};
   * if (comm.rank() == 0) in_vec = {0, 1, 2, 3, 4, 5, 6, 7};
   * auto out_vec = std::vector<int>(mpi::chunk_length(5, comm.size(), comm.rank()), 0);
   *
   * // scatter the middle elements of the input vector from rank 0 to all ranks
   * mpi::scatter_range(std::span{in_vec.data() + 1, 5}, out_vec, 5, comm);
   *
   * // output result
   * for (auto x : out_vec) std::cout << x << " ";
   * std::cout << std::endl;
   * @endcode
   *
   * Output (with 2 processes):
   *
   * ```
   * 4 5
   * 1 2 3
   * ```
   *
   * @tparam R1 mpi::contiguous_sized_range type.
   * @tparam R2 mpi::contiguous_sized_range type.
   * @param in_rg Range to scatter.
   * @param out_rg Range to scatter into.
   * @param in_size Size of the input range on root (must also be given on non-root ranks).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param chunk_size Size of the chunks to scatter.
   */
  template <contiguous_sized_range R1, contiguous_sized_range R2>
    requires(std::same_as<std::ranges::range_value_t<R1>, std::ranges::range_value_t<R2>>)
  void scatter_range(R1 &&in_rg, R2 &&out_rg, long in_size, communicator c = {}, int root = 0, long chunk_size = 1) {
    // check the sizes of the input and output ranges
    if (c.rank() == root) {
      EXPECTS_WITH_MESSAGE(in_size == std::ranges::size(in_rg), "Input range size not equal to provided size in mpi::scatter_range");
    }
    EXPECTS_WITH_MESSAGE(in_size == all_reduce(std::ranges::size(out_rg), c),
                         "Output range sizes don't add up to input range size in mpi::scatter_range");

    // do nothing if the input range is empty
    if (in_size == 0) return;

    // simply copy if there is no active MPI environment or if the communicator size is < 2
    if (!has_env || c.size() < 2) {
      std::ranges::copy(std::forward<R1>(in_rg), std::ranges::data(out_rg));
      return;
    }

    // check the size of the output range
    int recvcount = static_cast<int>(chunk_length(in_size, c.size(), c.rank(), chunk_size));
    EXPECTS_WITH_MESSAGE(recvcount == std::ranges::size(out_rg), "Output range size is incorrect in mpi::scatter_range");

    // prepare arguments for the MPI call
    auto sendcounts = std::vector<int>(c.size());
    auto displs     = std::vector<int>(c.size() + 1, 0);
    for (int i = 0; i < c.size(); ++i) {
      sendcounts[i] = static_cast<int>(chunk_length(in_size, c.size(), i, chunk_size));
      displs[i + 1] = sendcounts[i] + displs[i];
    }

    // scatter the range
    using in_value_t  = std::ranges::range_value_t<R1>;
    using out_value_t = std::ranges::range_value_t<R2>;
    if constexpr (has_mpi_type<in_value_t> && has_mpi_type<out_value_t>) {
      // make an MPI C library call for MPI compatible value types
      auto const in_data = std::ranges::data(in_rg);
      auto out_data      = std::ranges::data(out_rg);
      check_mpi_call(MPI_Scatterv(in_data, sendcounts.data(), displs.data(), mpi_type<in_value_t>::get(), out_data, recvcount,
                                  mpi_type<out_value_t>::get(), root, c.get()),
                     "MPI_Scatterv");
    } else {
      // otherwise throw an exception
      throw std::runtime_error{"Error in mpi::scatter_range: Types with no corresponding datatype can only be all-gathered"};
    }
  }

  /**
   * @brief Implementation of an MPI gather for an mpi::contiguous_sized_range.
   *
   * @details If mpi::has_mpi_type is true for the value type of the input ranges, then the ranges are gathered using a
   * simple `MPI_Gatherv` or `MPI_Allgatherv`. Otherwise, each process broadcasts its elements to all other processes
   * which implies that `all == true` is required in this case.
   *
   * It throws an exception in case a call to the MPI C library fails and it expects that the sizes of the input ranges
   * add up to the given size of the output range and that the output ranges have the correct size on receiving
   * processes.
   *
   * If the input ranges are all empty, it does nothing. If mpi::has_env is false or if the communicator size is < 2, it
   * simply copies the input range to the output range.
   *
   * @note It is recommended to use the generic mpi::gather for supported types, e.g. `std::vector` and `std::string`.
   * It is the user's responsibility to ensure that the ranges have the correct sizes.
   *
   * @code{.cpp}
   * // create input and output vectors on all ranks
   * auto in_vec  = std::vector<int>{0, 1, 2, 3, 4};
   * auto out_vec = std::vector<int>(3 * comm.size(), 0);
   *
   * // gather the middle elements of the input vectors from all ranks on rank 0
   * mpi::gather_range(std::span{in_vec.data() + 1, 3}, out_vec, 3 * comm.size(), comm);
   *
   * // output result
   * for (auto x : out_vec) std::cout << x << " ";
   * std::cout << std::endl;
   * @endcode
   *
   * Output (with 2 processes):
   *
   * ```
   * 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0
   * 1 2 3 1 2 3 1 2 3 1 2 3
   * ```
   *
   * @tparam R1 mpi::contiguous_sized_range type.
   * @tparam R2 mpi::contiguous_sized_range type.
   * @param in_rg Range to gather.
   * @param out_rg Range to gather into.
   * @param out_size Size of the output range on receiving processes (must also be given on non-receiving ranks).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   */
  template <contiguous_sized_range R1, contiguous_sized_range R2>
  void gather_range(R1 &&in_rg, R2 &&out_rg, long out_size, communicator c = {}, int root = 0, bool all = false) {
    // check the sizes of the input and output ranges
    auto const in_size = std::ranges::size(in_rg);
    EXPECTS_WITH_MESSAGE(out_size = all_reduce(in_size, c), "Input range sizes don't add up to output range size in mpi::gather_range");
    if (c.rank() == root || all) {
      EXPECTS_WITH_MESSAGE(out_size == std::ranges::size(out_rg), "Output range size is incorrect in mpi::gather_range");
    }

    // do nothing if the output range is empty
    if (out_size == 0) return;

    // simply copy if there is no active MPI environment or if the communicator size is < 2
    if (!has_env || c.size() < 2) {
      std::ranges::copy(std::forward<R1>(in_rg), std::ranges::data(out_rg));
      return;
    }

    // prepare arguments for the MPI call
    auto recvcounts = std::vector<int>(c.size());
    auto displs     = std::vector<int>(c.size() + 1, 0);
    int sendcount   = in_size;
    if (!all)
      check_mpi_call(MPI_Gather(&sendcount, 1, mpi_type<int>::get(), recvcounts.data(), 1, mpi_type<int>::get(), root, c.get()), "MPI_Gather");
    else
      check_mpi_call(MPI_Allgather(&sendcount, 1, mpi_type<int>::get(), recvcounts.data(), 1, mpi_type<int>::get(), c.get()), "MPI_Allgather");
    for (int i = 0; i < c.size(); ++i) displs[i + 1] = recvcounts[i] + displs[i];

    // gather the ranges
    using in_value_t  = std::ranges::range_value_t<R1>;
    using out_value_t = std::ranges::range_value_t<R2>;
    if constexpr (has_mpi_type<in_value_t> && has_mpi_type<out_value_t>) {
      // make an MPI C library call for MPI compatible value types
      auto const in_data = std::ranges::data(in_rg);
      auto out_data      = std::ranges::data(out_rg);
      if (!all)
        check_mpi_call(MPI_Gatherv(in_data, sendcount, mpi_type<in_value_t>::get(), out_data, recvcounts.data(), displs.data(),
                                   mpi_type<out_value_t>::get(), root, c.get()),
                       "MPI_Gatherv");
      else
        check_mpi_call(MPI_Allgatherv(in_data, sendcount, mpi_type<in_value_t>::get(), out_data, recvcounts.data(), displs.data(),
                                      mpi_type<out_value_t>::get(), c.get()),
                       "MPI_Allgatherv");
    } else {
      if (all) {
        // if all == true, each process broadcasts it elements to all other ranks
        for (int i = 0; i < c.size(); ++i) {
          auto view = std::views::drop(out_rg, displs[i]) | std::views::take(displs[i + 1] - displs[i]);
          if (c.rank() == i) std::ranges::copy(in_rg, std::ranges::begin(view));
          broadcast_range(view, c, i);
        }
      } else {
        // otherwise throw an exception
        throw std::runtime_error{"Error in mpi::gather_range: Types with no corresponding datatype can only be all-gathered"};
      }
    }
  }

} // namespace mpi
