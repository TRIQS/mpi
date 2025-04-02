// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides an MPI broadcast, reduce, scatter and gather for `std::vector`.
 */

#pragma once

#include "./communicator.hpp"
#include "./generic_communication.hpp"
#include "./ranges.hpp"
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

  /**
   * @brief Implementation of an MPI broadcast for a `std::vector`.
   *
   * @details It first broadcasts the size of the vector from the root process to all other processes, then resizes the
   * vector on all non-root processes and calls mpi::broadcast_range with the (resized) input vector.
   *
   * @tparam T Value type of the vector.
   * @param v `std::vector` to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> void mpi_broadcast(std::vector<T> &v, communicator c = {}, int root = 0) {
    auto count = v.size();
    broadcast(count, c, root);
    if (c.rank() != root) v.resize(count);
    broadcast_range(v, c, root);
  }

  /**
   * @brief Implementation of an MPI reduce for a `std::vector`.
   *
   * @details It first constructs the output vector with its value type equal to the return type of
   * `reduce(std::declval<T>())`. On receiving ranks, the output vector is then resized to the size of the input vector.
   * On non-receiving ranks, the output vector is always empty.
   *
   * It calls mpi::reduce_range with the input and constructed output vector.
   *
   * @tparam T Value type of the vector.
   * @param v `std::vector` to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return `std::vector` containing the result of the reduction.
   */
  template <typename T> auto mpi_reduce(std::vector<T> const &v, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    using value_type = std::remove_cvref_t<decltype(reduce(std::declval<T>()))>;
    std::vector<value_type> res(c.rank() == root || all ? v.size() : 0);
    reduce_range(v, res, c, root, all, op);
    return res;
  }

  /**
   * @brief Implementation of an MPI reduce for a `std::vector` that reduces directly into a given output vector.
   *
   * @details It first resizes the output vector to the size of the input vector on receiving ranks and then calls
   * mpi::reduce_range with the input and (resized) output vector.
   *
   * @tparam T1 Value type of the vector to be reduced.
   * @tparam T2 Value type of the vector to be reduced into.
   * @param v_in `std::vector` to reduce.
   * @param v_out `std::vector` to reduce into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T1, typename T2>
  void mpi_reduce_into(std::vector<T1> const &v_in, std::vector<T2> &v_out, communicator c = {}, int root = 0, bool all = false,
                       MPI_Op op = MPI_SUM) {
    if ((c.rank() == root || all) && v_out.size() != v_in.size()) v_out.resize(v_in.size());
    reduce_range(v_in, v_out, c, root, all, op);
  }

  /**
   * @brief Implementation of an MPI scatter for a `std::vector` that scatters directly into an existing output vector.
   *
   * @details It first broadcasts the size of the input vector from the root process to all other processes and
   * resizes the output vector if it has not the correct size. The size of the output vector is determined with
   * mpi::chunk_length. Then mpi::scatter_range is called with the input and (resized) output vector.
   *
   * @tparam T Value type of the vector.
   * @param v_in `std::vector` to scatter.
   * @param v_out `std::vector` to scatter into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> void mpi_scatter_into(std::vector<T> const &v_in, std::vector<T> &v_out, communicator c = {}, int root = 0) {
    auto scatter_size = static_cast<int>(v_in.size());
    broadcast(scatter_size, c, root);
    auto const recvcount = chunk_length(scatter_size, c.size(), c.rank());
    if (v_out.size() != recvcount) v_out.resize(recvcount);
    scatter_range(v_in, v_out, scatter_size, c, root);
  }

  /**
   * @brief Implementation of an MPI gather for a std::vector.
   *
   * @details It first all-reduces the sizes of the input vectors from all processes and then calls mpi::gather_range.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to gather.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result.
   * @return std::vector containing the result of the gather operation.
   */
  template <typename T> auto mpi_gather(std::vector<T> const &v, communicator c = {}, int root = 0, bool all = false) {
    long bsize = mpi::all_reduce(v.size(), c);
    std::vector<T> res(c.rank() == root || all ? bsize : 0);
    gather_range(v, res, bsize, c, root, all);
    return res;
  }

  /** @} */

} // namespace mpi
