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
 * @brief Provides a C++ wrapper class for an `MPI_Group` object.
 */

#pragma once

#include "./communicator.hpp"
#include "./environment.hpp"
#include "./utils.hpp"

#include <mpi.h>

#include <utility>
#include <vector>

namespace mpi {

  /**
   * @ingroup mpi_essentials
   * @brief C++ wrapper around `MPI_Group` providing various convenience functions.
   *
   * @details It stores an `MPI_Group` object as its only member which by default is set to `MPI_GROUP_NULL`.
   * The underlying `MPI_Group` object is automatically freed when a group object goes out of scope.
   *
   * This class follows move-only semantics and takes ownership of the wrapped `MPI_Group` object.
   *
   * All functions that make direct calls to the MPI C library throw an exception in case the call fails.
   */
  class group {
    public:
    /// Construct a group with `MPI_GROUP_NULL`.
    group() = default;

    /// Deleted copy constructor.
    group(group const &) = delete;

    /// Deleted copy assignment operator.
    group &operator=(group const &) = delete;

    /// Move constructor leaves moved-from object with `MPI_GROUP_NULL`.
    group(group &&other) noexcept : grp_{std::exchange(other.grp_, MPI_GROUP_NULL)} {}

    /// Move assignment operator leaves moved-from object with `MPI_GROUP_NULL`.
    group &operator=(group &&rhs) noexcept {
      if (this != std::addressof(rhs)) {
        free();
        grp_ = std::exchange(rhs.grp_, MPI_GROUP_NULL);
      }
      return *this;
    }

    /// Destructor calls free() to release the group.
    ~group() { free(); }

    /**
     * @brief Take ownership of an existing `MPI_Group` object.
     * @param grp `MPI_Group` to be handled.
     */
    explicit group(MPI_Group grp) : grp_(grp) {}

    /**
     * @brief Create a group from a communicator by calling `MPI_Comm_group`.
     * @param c mpi::communicator from which to create a group.
     */
    explicit group(communicator c) {
      if (has_env) check_mpi_call(MPI_Comm_group(c.get(), &grp_), "MPI_Comm_group");
    }

    /// Get the wrapped `MPI_Group` object.
    [[nodiscard]] MPI_Group get() const noexcept { return grp_; }

    /// Check if the contained `MPI_Group` is `MPI_GROUP_NULL`.
    [[nodiscard]] bool is_null() const noexcept { return grp_ == MPI_GROUP_NULL; }

    /**
     * @brief Get the rank of the calling process in the group.
     * @return The result of `MPI_Group_rank` if mpi::has_env is true, otherwise 0.
     */
    [[nodiscard]] int rank() const {
      int r = 0;
      if (has_env) check_mpi_call(MPI_Group_rank(grp_, &r), "MPI_Group_rank");
      return r;
    }

    /**
     * @brief Get the size of the group.
     * @return The result of `MPI_Group_size` if mpi::has_env is true, otherwise 1.
     */
    [[nodiscard]] int size() const {
      int s = 1;
      if (has_env) check_mpi_call(MPI_Group_size(grp_, &s), "MPI_Group_size");
      return s;
    }

    /**
     * @brief Create a new group by calling `MPI_Group_incl`.
     *
     * @details It produces a new group by reordering the existing group and taking only listed members.
     *
     * @param ranks List of ranks to include in the new group.
     * @return New group containing only the listed members.
     */
    [[nodiscard]] group include(std::vector<int> const &ranks) const {
      MPI_Group newgroup = MPI_GROUP_NULL;
      if (has_env) check_mpi_call(MPI_Group_incl(grp_, static_cast<int>(ranks.size()), ranks.data(), &newgroup), "MPI_Group_incl");
      return group{newgroup};
    }

    /// Free the group by calling `MPI_Group_free` (if it is not is_null()).
    void free() noexcept {
      if (has_env && !is_null()) MPI_Group_free(&grp_);
    }

    private:
    MPI_Group grp_ = MPI_GROUP_NULL;
  };

} // namespace mpi
