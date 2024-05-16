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

#include "./mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Implementation of an MPI broadcast for a std::vector.
   *
   * @details If mpi::has_mpi_type<T> is true then the vector is broadcasted using a simple `MPI_Bcast`. Otherwise,
   * the generic mpi::broadcast is called for each element of the vector.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> void mpi_broadcast(std::vector<T> &v, communicator c = {}, int root = 0) {
    auto s = v.size();
    broadcast(s, c, root);
    if (c.rank() != root) v.resize(s);
    if constexpr (has_mpi_type<T>) {
      if (s != 0) MPI_Bcast(v.data(), v.size(), mpi_type<T>::get(), root, c.get());
    } else {
      for (auto &x : v) broadcast(x, c, root);
    }
  }

  /**
   * @brief Implementation of an in-place MPI reduce for a std::vector.
   *
   * @details If mpi::has_mpi_type<T> is true then the vector is reduced using a simple `MPI_Reduce` or `MPI_Allreduce`.
   * Otherwise, the specialized `mpi_reduce_in_place` is called for each element of the vector.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T> void mpi_reduce_in_place(std::vector<T> &v, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    if (v.size() == 0) return;
    if constexpr (has_mpi_type<T>) {
      if (!all)
        MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : v.data()), v.data(), v.size(), mpi_type<T>::get(), op, root, c.get());
      else
        MPI_Allreduce(MPI_IN_PLACE, v.data(), v.size(), mpi_type<T>::get(), op, c.get());
    } else {
      for (auto &x : v) mpi_reduce_in_place(v, c, root, all);
    }
  }

  namespace detail {

    // Helper struct to get the regular type of a type.
    template <typename T, typename Enable = void> struct _regular {
      using type = T;
    };

    // Spezialization of _regular for types with a `regular_type` type alias.
    template <typename T> struct _regular<T, std::void_t<typename T::regular_type>> {
      using type = typename T::regular_type;
    };

  } // namespace detail

  /**
   * @ingroup utilities
   * @brief Type trait to get the regular type of a type.
   * @tparam T Type to check.
   */
  template <typename T> using regular_t = typename detail::_regular<std::decay_t<T>>::type;

  /**
   * @brief Implementation of an MPI reduce for a std::vector.
   *
   * @details If mpi::has_mpi_type<T> is true then the vector is reduced using a simple `MPI_Reduce` or `MPI_Allreduce`
   * (in this case, mpi::regular_t<T> has to be the same as T). Otherwise, the generic mpi::reduce is called for each
   * element of the vector.
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
    auto s = v.size();

    // check if all vectors are of the same size, otherwise abort
    if (all) {
      auto max_size = mpi_reduce(s, c, root, all, MPI_MAX);
      if (s != max_size) {
        std::cerr << "Cannot all_reduce vectors of different sizes\n";
        std::abort();
      }
    }

    // return an empty vector if size is 0
    if (s == 0) return {};

    // perform the reduction for every element of the vector
    if constexpr (has_mpi_type<T>) {
      static_assert(std::is_same_v<regular_t<T>, T>, "Internal error");
      std::vector<T> res(s);
      if (!all)
        MPI_Reduce((void *)v.data(), res.data(), s, mpi_type<T>::get(), op, root, c.get());
      else
        MPI_Allreduce((void *)v.data(), res.data(), s, mpi_type<T>::get(), op, c.get());
      return res;
    } else {
      std::vector<regular_t<T>> r;
      r.reserve(s);
      for (size_t i = 0; i < s; ++i) r.push_back(reduce(v[i], c, root, all, op));
      return r;
    }
  }

  /**
   * @brief Implementation of an MPI scatter for a std::vector.
   *
   * @details If mpi::has_mpi_type<T> is true then the vector is scattered as evenly as possible across the processes
   * in the communicator using a simple `MPI_Scatterv`.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to scatter.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @return std::vector containing the result of the scatter operation.
   */
  template <typename T> std::vector<T> mpi_scatter(std::vector<T> const &v, communicator c = {}, int root = 0) {
    auto s = v.size();

    // return an empty vector if size is 0
    if (s == 0) return {};

    // arguments for the MPI call
    auto sendcounts = std::vector<int>(c.size());          // number of elements sent to each process
    auto displs     = std::vector<int>(c.size() + 1, 0);   // displacements given in number of elements not in bytes
    int recvcount   = chunk_length(s, c.size(), c.rank()); // number of elements received by the calling process
    for (int r = 0; r < c.size(); ++r) {
      sendcounts[r] = chunk_length(s, c.size(), r);
      displs[r + 1] = sendcounts[r] + displs[r];
    }

    // do the scattering
    std::vector<T> res(recvcount);
    if constexpr (has_mpi_type<T>) {
      MPI_Scatterv((void *)v.data(), &sendcounts[0], &displs[0], mpi_type<T>::get(), (void *)res.data(), recvcount, mpi_type<T>::get(), root,
                   c.get());
    } else {
      std::copy(cbegin(v) + displs[c.rank()], cbegin(v) + displs[c.rank() + 1], begin(res));
    }

    return res;
  }

  /**
   * @brief Implementation of an MPI gather for a std::vector.
   *
   * @details If mpi::has_mpi_type<T> is true then the vector is gathered using a simple `MPI_Gatherv` or `MPI_Allgatherv`.
   * Otherwise, each process broadcasts its elements to all other processes which implies that `all == true` is required
   * in this case.
   *
   * @tparam T Value type of the vector.
   * @param v std::vector to gather.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @return std::vector containing the result of the gather operation.
   */
  template <typename T> std::vector<T> mpi_gather(std::vector<T> const &v, communicator c = {}, int root = 0, bool all = false) {
    long s = mpi_reduce(v.size(), c, root, all);

    // return an empty vector if size is 0
    if (s == 0) return {};

    // arguments for the MPI call
    auto mpi_ty     = mpi_type<int>::get();
    auto recvcounts = std::vector<int>(c.size());        // number of elements received from each process
    auto displs     = std::vector<int>(c.size() + 1, 0); // displacements given in number of elements not in bytes
    int sendcount   = v.size();                          // number of elements sent by the calling process
    if (!all)
      MPI_Gather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, root, c.get());
    else
      MPI_Allgather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, c.get());

    for (int r = 0; r < c.size(); ++r) displs[r + 1] = recvcounts[r] + displs[r];

    // do the gathering
    std::vector<T> res((all || (c.rank() == root) ? s : 0));
    if constexpr (has_mpi_type<T>) {
      if (!all)
        MPI_Gatherv((void *)v.data(), sendcount, mpi_type<T>::get(), (void *)res.data(), &recvcounts[0], &displs[0], mpi_type<T>::get(), root,
                    c.get());
      else
        MPI_Allgatherv((void *)v.data(), sendcount, mpi_type<T>::get(), (void *)res.data(), &recvcounts[0], &displs[0], mpi_type<T>::get(), c.get());
    } else {
      if (!all)
        throw std::runtime_error{"mpi_gather for custom types only implemented with 'all = true'\n"};
      else {
        for (int r = 0; r < c.size(); ++r) {
          for (auto i = displs[r]; i < displs[r + 1]; ++i) {
            if (c.rank() == r) res[i] = v[i - displs[r]];
            mpi::broadcast(res[i], c, r);
          }
        }
      }
    }
    return res;
  }

  /** @} */

} // namespace mpi
