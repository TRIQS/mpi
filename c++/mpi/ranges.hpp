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
 * @brief Provides an MPI broadcast, reduce, scatter and gather for generic ranges.
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
#include <concepts>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Implementation of an MPI broadcast for `std::ranges::sized_range` objects.
   *
   * @details The behaviour of this function is as follows:
   * - If the number of elements to be broadcasted is zero, it does nothing.
   * - If the range is contiguous with an MPI compatible value type, it calls `MPI_Bcast` and broadcasts the elements
   * from the input range on the root process to all other processes.
   * - Otherwise, it calls mpi::broadcast for each element separately.
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call and it expects that the input range 
   * size is equal on all processes.
   *
   * @tparam R `std::ranges::sized_range` type.
   * @param rg Range to be broadcasted (into).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <std::ranges::sized_range R> void broadcast_range(R &&rg, communicator c = {}, int root = 0) { // NOLINT (ranges need not be forwarded)
    // check the size of the range
    auto size = static_cast<long>(std::ranges::size(rg));
    EXPECTS_WITH_MESSAGE(all_equal(size, c), "Range sizes are not equal on all processes in mpi::broadcast_range");

    // do nothing if no elements are broadcasted
    if (size <= 0) return;

    // call the MPI C library if the ranges are contiguous with MPI compatible value types, otherwise do element-wise
    // broadcasts
    if constexpr (MPICompatibleRange<R>) {
      // in case there is no active MPI environment or if the communicator size is < 2, do nothing
      if (!has_env || c.size() < 2) return;

      // make the MPI C library call (allow the number of elements to larger than INT_MAX)
      constexpr long max_int = std::numeric_limits<int>::max();
      for (long offset = 0; size > 0; offset += max_int, size -= max_int) {
        auto const count = static_cast<int>(std::min(size, max_int));
        check_mpi_call(MPI_Bcast(std::ranges::data(rg) + offset, count, mpi_type<std::ranges::range_value_t<R>>::get(), root, c.get()), "MPI_Bcast");
      }
    } else {
      // otherwise call the generic broadcast for each element separately
      for (auto &x : rg) broadcast(x, c, root);
    }
  }

  /**
   * @brief Implementation of an MPI reduce for `std::ranges::sized_range` objects.
   *
   * @details The behaviour of this function is as follows:
   * - If the number of elements to be reduced is zero, it does nothing.
   * - If the range is contiguous with an MPI compatible value type, it calls `MPI_Reduce` or `MPI_Allreduce` to reduce
   * the elements in the input ranges into the output ranges on receiving ranks.
   *   - If the input and output ranges point to the same data, the reduction is done in place.
   * - Otherwise, it calls mpi::reduce_into for each input-output element pair separately.
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call and it expects
   * - that the input range size on all processes and the output range size on receiving processes are equal and
   * - that either all or none of the receiving processes choose the in place option.
   *
   * @tparam R1 `std::ranges::sized_range` type.
   * @tparam R2 `std::ranges::sized_range` type.
   * @param in_rg Range to be reduced.
   * @param out_rg Range to be reduced into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <std::ranges::sized_range R1, std::ranges::sized_range R2>
  void reduce_range(R1 &&in_rg, R2 &&out_rg, communicator c = {}, int root = 0, bool all = false, // NOLINT (ranges need not be forwarded)
                    MPI_Op op = MPI_SUM) {
    // check the size of the input range
    auto size = static_cast<long>(std::ranges::size(in_rg));
    EXPECTS_WITH_MESSAGE(all_equal(size, c), "Input range sizes are not equal on all processes in mpi::reduce_range");

    // do nothing if no elements are reduced
    if (size <= 0) return;

    // check the size of the output range
    bool const receives = (c.rank() == root || all);
    if (receives) EXPECTS_WITH_MESSAGE(size == std::ranges::size(out_rg), "Input and output range sizes are not equal in mpi::reduce_range");

    // call the MPI C library if the ranges are contiguous with MPI compatible value types
    if constexpr (MPICompatibleRange<R1> && MPICompatibleRange<R2>) {
      static_assert(std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R1>>, std::remove_cvref_t<std::ranges::range_value_t<R2>>>,
                    "Value types of input and output ranges not compatible in mpi::reduce_range");

      // check if the reduction is in place
      bool const in_place = (static_cast<void const *>(std::ranges::data(in_rg)) == static_cast<void *>(std::ranges::data(out_rg)));
      if (all) {
        EXPECTS_WITH_MESSAGE(all_equal(static_cast<int>(in_place), c),
                             "Either zero or all receiving processes have to choose the in place option in mpi::reduce_range");
      }

      // in case there is no active MPI environment or if the communicator size is < 2, copy to the output range
      if (!has_env || c.size() < 2) {
        std::ranges::copy(std::forward<R1>(in_rg), std::ranges::data(out_rg));
        return;
      }

      // make the MPI C library call (allow the number of elements to larger than INT_MAX)
      constexpr long max_int = std::numeric_limits<int>::max();
      for (long offset = 0; size > 0; offset += max_int, size -= max_int) {
        auto in_data  = static_cast<void const *>(std::ranges::data(in_rg) + offset);
        auto out_data = std::ranges::data(out_rg) + offset;
        if (receives and in_place) in_data = MPI_IN_PLACE;
        auto const count = static_cast<int>(std::min(size, max_int));
        if (all) {
          check_mpi_call(MPI_Allreduce(in_data, out_data, count, mpi_type<std::ranges::range_value_t<R1>>::get(), op, c.get()), "MPI_Allreduce");
        } else {
          check_mpi_call(MPI_Reduce(in_data, out_data, count, mpi_type<std::ranges::range_value_t<R1>>::get(), op, root, c.get()), "MPI_Reduce");
        }
      }
    } else {
      // fallback to element-wise reduction if the range is not contiguous with an MPI compatible value type
      if (size <= std::ranges::size(out_rg)) {
        // on ranks where the output range size is large enough, reduce into the output elements
        for (auto &&[x_in, x_out] : itertools::zip(in_rg, out_rg)) reduce_into(x_in, x_out, c, root, all, op);
      } else {
        // on all other ranks, reduce into a dummy output object (needs to be default constructible)
        using out_value_t = std::ranges::range_value_t<R2>;
        if constexpr (std::is_default_constructible_v<out_value_t>) {
          out_value_t out_dummy{};
          for (auto &&x_in : in_rg) reduce_into(x_in, out_dummy, c, root, all, op);
        } else {
          // if it is not default constructible, is there something we can do?
          throw std::runtime_error("Cannot default construct dummy object in mpi::reduce_range");
        }
      }
    }
  }

  /**
   * @brief Implementation of an MPI scatter for mpi::MPICompatibleRange objects.
   *
   * @details The behaviour of this function is as follows:
   * - If the number of elements to be scattered is zero, it does nothing.
   * - Otherwise, it calls `MPI_Scatterv` to scatter the input range from the root process to the output ranges on all
   * other processes.
   *
   * By default, the input range is scattered as evenly as possible from the root process to all other processes in the
   * communicator. To change that, the user can specify a chunk size which is used to divide the number of elements to
   * be scattered into chunks of the specified size. Then, instead of single elements, the chunks are distributed evenly
   * across the processes in the communicator.
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call and it expects
   * - that the number of elements to be scattered is equal on all processes,
   * - that the size of the input range on the root process is equal the number of elements to be scattered and
   * - that the output range size is equal the number of elements to be received on all processes.
   *
   * @note In place scattering is not supported.
   *
   * @tparam R1 mpi::MPICompatibleRange type.
   * @tparam R2 mpi::MPICompatibleRange type.
   * @param in_rg Range to be scattered.
   * @param out_rg Range to be scattered into.
   * @param scatter_size Number of elements to be scattered.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param chunk_size Size of the chunks to scatter.
   */
  template <MPICompatibleRange R1, MPICompatibleRange R2>
    requires(std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R1>>, std::remove_cvref_t<std::ranges::range_value_t<R2>>>)
  void scatter_range(R1 &&in_rg, R2 &&out_rg, long scatter_size, communicator c = {}, int root = 0, // NOLINT (ranges need not be forwarded)
                     long chunk_size = 1) {
    // check the number of elements to be scattered
    EXPECTS_WITH_MESSAGE(all_equal(scatter_size, c), "Number of elements to be scattered is not equal on all processes in mpi::scatter_range");

    // do nothing if no elements are scattered
    if (scatter_size == 0) return;

    // check the size of the input range on root
    if (c.rank() == root) {
      EXPECTS_WITH_MESSAGE(scatter_size == std::ranges::size(in_rg),
                           "Input range size on root is not equal the number of elements to be scattered in mpi::scatter_range");
    }

    // check the size of the output range
    auto const recvcount = static_cast<int>(chunk_length(scatter_size, c.size(), c.rank(), chunk_size));
    EXPECTS_WITH_MESSAGE(recvcount == std::ranges::size(out_rg),
                         "Output range size is not equal the number of elements to be received in mpi::scatter_range");

    // in case there is no active MPI environment or if the communicator size is < 2, copy to output range
    if (!has_env || c.size() < 2) {
      std::ranges::copy(std::forward<R1>(in_rg), std::ranges::data(out_rg));
      return;
    }

    // prepare arguments for the MPI call
    auto sendcounts = std::vector<int>(c.size());
    auto displs     = std::vector<int>(c.size() + 1, 0);
    for (int i = 0; i < c.size(); ++i) {
      sendcounts[i] = static_cast<int>(chunk_length(scatter_size, c.size(), i, chunk_size));
      displs[i + 1] = sendcounts[i] + displs[i];
    }

    // make the MPI C library call
    check_mpi_call(MPI_Scatterv(std::ranges::data(in_rg), sendcounts.data(), displs.data(), mpi_type<std::ranges::range_value_t<R1>>::get(),
                                std::ranges::data(out_rg), recvcount, mpi_type<std::ranges::range_value_t<R2>>::get(), root, c.get()),
                   "MPI_Scatterv");
  }

  /**
   * @brief Implementation of an MPI gather for mpi::MPICompatibleRange objects.
   *
   * @details The behaviour of this function is as follows:
   * - If the number of elements to be gathered is zero, it does nothing.
   * - Otherwise, it calls `MPI_Gatherv` or `MPI_Allgatherv` to gather the elements from the input ranges on all
   * processes into the output ranges on receiving processes.
   *
   * This is the inverse operation of mpi::scatter_range. The numbers of elements to be gathered do not have to be equal
   * on all processes.
   *
   * Direct calls the MPI C API are checked for success with mpi::check_mpi_call and it expects that the output range 
   * sizes on receiving processes is the number of elements to be gathered.
   *
   * @note In place gathering is not supported.
   *
   * @tparam R1 mpi::MPICompatibleRange type.
   * @tparam R2 mpi::MPICompatibleRange type.
   * @param in_rg Range to be gathered.
   * @param out_rg Range to be gathered into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather operation.
   */
  template <MPICompatibleRange R1, MPICompatibleRange R2>
    requires(std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R1>>, std::remove_cvref_t<std::ranges::range_value_t<R2>>>)
  void gather_range(R1 &&in_rg, R2 &&out_rg, communicator c = {}, int root = 0, bool all = false) { // NOLINT (ranges need not be forwarded)
    // get the receive counts (sendcount from each process) and the displacements
    auto sendcount  = static_cast<int>(std::ranges::size(in_rg));
    auto recvcounts = all_gather(sendcount, c);
    auto displs     = std::vector<int>(c.size() + 1, 0);
    std::partial_sum(recvcounts.begin(), recvcounts.end(), displs.begin() + 1);

    // do nothing if there are no elements to gather
    if (displs.back() == 0) return;

    // check the size of the output range on receiving ranks
    if (c.rank() == root || all) {
      EXPECTS_WITH_MESSAGE(displs.back() == std::ranges::size(out_rg),
                           "Output range size is not equal the number of elements to be received in mpi::gather_range");
    }

    // in case there is no active MPI environment or if the communicator size is < 2, copy to the output range
    if (!has_env || c.size() < 2) {
      std::ranges::copy(std::forward<R1>(in_rg), std::ranges::data(out_rg));
      return;
    }

    // make the MPI C library call
    if (all) {
      check_mpi_call(MPI_Allgatherv(std::ranges::data(in_rg), sendcount, mpi_type<std::ranges::range_value_t<R1>>::get(), std::ranges::data(out_rg),
                                    recvcounts.data(), displs.data(), mpi_type<std::ranges::range_value_t<R2>>::get(), c.get()),
                     "MPI_Allgatherv");
    } else {
      check_mpi_call(MPI_Gatherv(std::ranges::data(in_rg), sendcount, mpi_type<std::ranges::range_value_t<R1>>::get(), std::ranges::data(out_rg),
                                 recvcounts.data(), displs.data(), mpi_type<std::ranges::range_value_t<R2>>::get(), root, c.get()),
                     "MPI_Gatherv");
    }
  }

  /** @} */

} // namespace mpi
