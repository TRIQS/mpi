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
#include "mpi.hpp"
#include <vector>

namespace mpi {

  // ---------------- broadcast ---------------------

  template <typename T>
  void mpi_broadcast(std::vector<T> &v, communicator c = {}, int root = 0) {
    size_t s = v.size();
    broadcast(s, c, root);
    if (c.rank() != root) v.resize(s);
    if constexpr (has_mpi_type<T>) {
      if (s != 0) MPI_Bcast(v.data(), v.size(), mpi_type<T>::get(), root, c.get());
    } else {
      for (auto &x : v) broadcast(x, c, root);
    }
  }
  // ---------------- reduce in place  ---------------------

  template <typename T>
  void mpi_reduce_in_place(std::vector<T> &a, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    if (a.size() == 0) return; // mpi behaviour not checked in that case.
    if constexpr (has_mpi_type<T>) {
      if (!all)
        MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : a.data()), a.data(), a.size(), mpi_type<T>::get(), op, root, c.get());
      else
        MPI_Allreduce(MPI_IN_PLACE, a.data(), a.size(), mpi_type<T>::get(), op, c.get());
    } else {
      for (auto &x : a) mpi_reduce_in_place(a, c, root, all);
    }
  }

  // ---------------- reduce   ---------------------

  namespace detail {
    template <typename T, typename Enable = void>
    struct _regular {
      using type = T;
    };
    template <typename T>
    struct _regular<T, std::void_t<typename T::regular_type>> {
      using type = typename T::regular_type;
    };
  } // namespace detail

  template <typename T>
  using regular_t = typename detail::_regular<std::decay_t<T>>::type;

  template <typename T>
  std::vector<regular_t<T>> mpi_reduce(std::vector<T> const &a, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    size_t s = a.size();

    // For all==true, guarantee that all vectors are of the same size
    if (all) {
      auto max_size = mpi_reduce(s, c, root, all, MPI_MAX);
      if (s != max_size) {
        std::cerr << "Cannot all_reduce vectors of different sizes\n";
        std::abort();
      }
    }

    if (s == 0) return {}; // nothing to do, and MPI does not like size 0
    if constexpr (has_mpi_type<T>) {
      static_assert(std::is_same_v<regular_t<T>, T>, "Internal error");
      std::vector<T> r(s);
      if (!all)
        MPI_Reduce((void *)a.data(), r.data(), s, mpi_type<T>::get(), op, root, c.get());
      else
        MPI_Allreduce((void *)a.data(), r.data(), s, mpi_type<T>::get(), op, c.get());
      return r;
    } else {
      std::vector<regular_t<T>> r;
      r.reserve(s);
      for (size_t i = 0; i < s; ++i) r.push_back(reduce(a[i], c, root, all, op));
      return r;
    }
  }

  // ---------------- scatter  ---------------------

  // FIXME : not checked for 0 size ?
  template <typename T>
  std::vector<T> mpi_scatter(std::vector<T> const &a, communicator c = {}, int root = 0) {

    auto slow_size  = a.size();
    auto sendcounts = std::vector<int>(c.size());
    auto displs     = std::vector<int>(c.size() + 1, 0);
    int recvcount   = chunk_length(slow_size, c.size(), c.rank());
    std::vector<T> b(recvcount);

    for (int r = 0; r < c.size(); ++r) {
      sendcounts[r] = chunk_length(slow_size, c.size(), r);
      displs[r + 1] = sendcounts[r] + displs[r];
    }

    if constexpr (has_mpi_type<T>) {
      MPI_Scatterv((void *)a.data(), &sendcounts[0], &displs[0], mpi_type<T>::get(), (void *)b.data(), recvcount, mpi_type<T>::get(), root, c.get());
    } else {
      std::copy(cbegin(a) + displs[c.rank()], cbegin(a) + displs[c.rank() + 1], begin(b));
    }

    return b;
  }
  // ---------------- gather  ---------------------

  template <typename T>
  std::vector<T> mpi_gather(std::vector<T> const &a, communicator c = {}, int root = 0, bool all = false) {

    long size = mpi_reduce(a.size(), c, root, all);
    std::vector<T> b((all || (c.rank() == root) ? size : 0));

    auto recvcounts = std::vector<int>(c.size());
    auto displs     = std::vector<int>(c.size() + 1, 0);
    int sendcount   = a.size();
    auto mpi_ty     = mpi_type<int>::get();
    if (!all)
      MPI_Gather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, root, c.get());
    else
      MPI_Allgather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, c.get());

    for (int r = 0; r < c.size(); ++r) displs[r + 1] = recvcounts[r] + displs[r];

    if constexpr (has_mpi_type<T>) {
      if (!all)
        MPI_Gatherv((void *)a.data(), sendcount, mpi_type<T>::get(), (void *)b.data(), &recvcounts[0], &displs[0], mpi_type<T>::get(), root, c.get());
      else
        MPI_Allgatherv((void *)a.data(), sendcount, mpi_type<T>::get(), (void *)b.data(), &recvcounts[0], &displs[0], mpi_type<T>::get(), c.get());
    } else {
      if (!all)
        throw std::runtime_error{"mpi_gather for custom types only implemented with 'all = true'\n"};
      else {
        for (int r = 0; r < c.size(); ++r) {
          for (auto i = displs[r]; i < displs[r + 1]; ++i) {
            if (c.rank() == r) b[i] = a[i - displs[r]];
            mpi::broadcast(b[i], c, r);
          }
        }
      }
    }

    return b;
  }
} // namespace mpi
