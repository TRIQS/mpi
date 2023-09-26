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
 * @brief Provides a C++ wrapper class for an @p MPI_Group object.
 */

#pragma once

#include "./communicator.hpp"
#include "./environment.hpp"

#include <mpi.h>

#include <cstdlib>
#include <unistd.h>

namespace mpi {

  /**
   * @ingroup mpi_essentials
   * @brief C++ wrapper around @p MPI_Group providing various convenience functions.
   *
   * @details It stores an @p MPI_Group object as its only member which by default is set to @p MPI_GROUP_NULL.
   */
  class group {
    // Wrapped @p MPI_Group object.
    MPI_Group _grp = MPI_GROUP_NULL;

    public:
    /// Construct a group with @p MPI_GROUP_NULL.
    group() = default;

    /// Deleted copy constructor.
    group(group const &) = delete;

    /// Deleted copy assignment operator.
    group &operator=(group const &) = delete;

    /// Move constructor leaves moved-from object with @p MPI_GROUP_NULL.
    group(group &&other) noexcept : _grp{std::exchange(other._grp, MPI_GROUP_NULL)} {}

    /// Move assignment operator leaves moved-from object with @p MPI_GROUP_NULL.
    group &operator=(group &&rhs) noexcept {
      if (this != std::addressof(rhs)) {
        this->free();
        this->_grp = std::exchange(rhs._grp, MPI_GROUP_NULL);
      }
      return *this;
    }

    /// Destructor
    virtual ~group() { free(); }

    /**
     * @brief Take ownership of an existing @p MPI_Group object.
     * @param grp The group to be handled.
     */
    explicit group(MPI_Group grp) : _grp(grp) {}

    /**
     * @brief Create a group from a communicator.
     * @param c The communicator from which to create a group.
     */
    explicit group(communicator c) {
      if (has_env) { MPI_Comm_group(c.get(), &_grp); }
    }

    /// Get the wrapped @p MPI_Group object.
    [[nodiscard]] MPI_Group get() const noexcept { return _grp; }

    /// Check if the contained @p MPI_Group is @p MPI_GROUP_NULL.
    [[nodiscard]] bool is_null() const noexcept { return _grp == MPI_GROUP_NULL; }

    /// Rank of the calling process in the given group.
    [[nodiscard]] int rank() const {
      int rank = 0;
      if (has_env) { MPI_Group_rank(_grp, &rank); }
      return rank;
    }

    /// Size of a group.
    [[nodiscard]] int size() const {
      int size = 1;
      if (has_env) { MPI_Group_size(_grp, &size); }
      return size;
    }

    /**
     * @brief Produces a group by reordering an existing group and taking only listed members.
     * @param ranks List of ranks to include in the new group.
     * @return New group containing only the listed members.
     */
    group include(std::vector<int> const &ranks) const {
      MPI_Group newgroup = MPI_GROUP_NULL;
      if (has_env) { MPI_Group_incl(_grp, ranks.size(), ranks.data(), &newgroup); }
      return group(newgroup);
    }

    /// Free the group.
    void free() {
      if (has_env) {
        if (_grp != MPI_GROUP_NULL) { MPI_Group_free(&_grp); }
      }
    }
  };

} // namespace mpi
