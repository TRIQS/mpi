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
 * @brief Provides an MPI broadcast, reduce, scatter and gather for std::vector.
 */

#pragma once

#include "./communicator.hpp"
#include "./generic_communication.hpp"
#include "./ranges.hpp"
#include "./utils.hpp"

#include <mpi.h>

#include <vector>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Implementation of an MPI broadcast for a std::vector.
   *
   * @details It first broadcasts the size of the vector from the root process to all other processes, then resizes the
   * vector on all non-root processes and calls mpi::broadcast_range with the (resized) input vector.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> void mpi_broadcast(std::vector<T> &v, communicator c = {}, int root = 0) {
    auto bsize = v.size();
    broadcast(bsize, c, root);
    if (c.rank() != root) v.resize(bsize);
    broadcast_range(v, c, root);
  }

  /**
   * @brief Implementation of an in-place MPI reduce for a std::vector.
   *
   * @details It simply calls mpi::reduce_in_place_range with the given input vector.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T> void mpi_reduce_in_place(std::vector<T> &v, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    reduce_in_place_range(v, c, root, all, op);
  }

  /**
   * @brief Implementation of an MPI reduce for a std::vector.
   *
   * @details It simply calls mpi::reduce_range with the given input vector and an empty vector of the same size.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return std::vector containing the result of each individual reduction.
   */
  template <typename T>
  std::vector<regular_t<T>> mpi_reduce(std::vector<T> const &v, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    std::vector<regular_t<T>> res(c.rank() == root || all ? v.size() : 0);
    reduce_range(v, res, c, root, all, op);
    return res;
  }

  /**
   * @brief Implementation of an MPI scatter for a std::vector.
   *
   * @details It first broadcasts the size of the vector from the root process to all other processes and then calls
   * mpi::scatter_range.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to scatter.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @return std::vector containing the result of the scatter operation.
   */
  template <typename T> std::vector<T> mpi_scatter(std::vector<T> const &v, communicator c = {}, int root = 0) {
    auto bsize = v.size();
    broadcast(bsize, c, root);
    std::vector<T> res(chunk_length(bsize, c.size(), c.rank()));
    scatter_range(v, res, bsize, c, root);
    return res;
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
  template <typename T> std::vector<T> mpi_gather(std::vector<T> const &v, communicator c = {}, int root = 0, bool all = false) {
    long bsize = mpi::all_reduce(v.size(), c);
    std::vector<T> res(c.rank() == root || all ? bsize : 0);
    gather_range(v, res, bsize, c, root, all);
    return res;
  }

  /** @} */

} // namespace mpi
