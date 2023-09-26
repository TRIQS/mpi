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
 * @brief Provides a C++ wrapper class for the @p MPI_Win object.
 */

#pragma once

#include "./communicator.hpp"
#include "./datatypes.hpp"
#include "./group.hpp"
#include "./macros.hpp"

#include <mpi.h>

#include <utility>

namespace mpi {

  /**
   * @addtogroup mpi_osc_shm
   * @{
   */

  template <class BaseType> class shared_window;

  /**
  * @brief A C++ wrapper around @p MPI_Win providing convenient memory window management.
  *
  * @details This class abstracts the complexities of MPI window management, allowing processes
  *          in an MPI communicator to create and share memory regions efficiently. It supports
  *          both local buffer-based windows and dynamically allocated memory windows.
  *
  *          If a base pointer is not specified, the constructor will allocate memory internally.
  *
  * @tparam BaseType The type of elements stored in the memory window.
  */

  template <class BaseType> class window {
    friend class shared_window<BaseType>;
    MPI_Win _win{MPI_WIN_NULL};
    communicator _comm{MPI_COMM_NULL};
    bool _owned{true};
    BaseType *_data{nullptr};
    MPI_Aint _size{0};

    public:
    window()               = default;
    window(window const &) = delete;
    window(window &&other) noexcept
       : _win{std::exchange(other._win, MPI_WIN_NULL)},
         _owned{std::exchange(other._owned, true)},
         _data{std::exchange(other._data, nullptr)},
         _size{std::exchange(other._size, 0)} {}
    window &operator=(window const &) = delete;
    window &operator=(window &&rhs) noexcept {
      if (this != std::addressof(rhs)) {
        this->free();
        this->_win   = std::exchange(rhs._win, MPI_WIN_NULL);
        this->_comm  = std::exchange(rhs._comm, communicator(MPI_COMM_NULL));
        this->_owned = std::exchange(rhs._owned, true);
        this->_data  = std::exchange(rhs._data, nullptr);
        this->_size  = std::exchange(rhs._size, 0);
      }
      return *this;
    }

    /**
    * @brief Constructs an MPI window over an existing local memory buffer.
    *
    * @details This constructor allows creating a window using a pre-allocated memory buffer.
    *          The window provides access to the specified memory region across MPI processes
    *          within the given communicator. The buffer is not freed upon destruction.
    *
    * @param c The MPI communicator that defines the group of processes sharing the window.
    * @param base Pointer to the base address of the memory buffer.
    * @param size The number of elements of type @p BaseType in the buffer. (default @p 0)
    * @param info Additional MPI information. (default @p MPI_INFO_NULL)
    */
    explicit window(communicator const &c, BaseType *base, MPI_Aint size = 0, MPI_Info info = MPI_INFO_NULL) noexcept(false) : _comm(c.get()) {
      ASSERT(size >= 0)
      ASSERT(!(base == nullptr && size > 0))
      if (has_env) {
        MPI_Win_create(base, size * sizeof(BaseType), sizeof(BaseType), info, c.get(), &_win);
        _data = base;
        _size = size;
      } else {
        _owned = false;
        _data  = base;
        _size  = size;
      }
    }

    /**
    * @brief Constructs an MPI window with dynamically allocated memory.
    *
    * @details This constructor allocates a new memory buffer locally and creates an MPI window
    *          over it. The allocated memory is automatically freed when the window is destroyed.
    *          This is useful when the memory region is meant to be shared across processes
    *          without needing an external buffer.
    *
    * @param c The MPI communicator that defines the group of processes sharing the window.
    * @param size The number of elements of type @p BaseType to allocate. (default @p 0)
    * @param info Additional MPI information. (default @p MPI_INFO_NULL)
    */
    explicit window(communicator const &c, MPI_Aint size = 0, MPI_Info info = MPI_INFO_NULL) noexcept : _comm(c.get()) {
      ASSERT(size >= 0)
      if (has_env) {
        void *baseptr = nullptr;
        MPI_Win_allocate(size * sizeof(BaseType), sizeof(BaseType), info, c.get(), &baseptr, &_win);
        _data = static_cast<BaseType *>(baseptr);
        _size = size;
      } else {
        _owned = true;
        _data  = new BaseType[size];
        _size  = size;
      }
    }

    /**
    * @brief Destroys the window and releases allocated resources.
    *
    * Before freeing, a window must have completed all its involvement in RMA
    * communications.  For that reason the destructor implicitly calls @p
    * fence().  The window also must be unlocked if it has been previously
    * locked, however, this cannot be detected and is therefore the
    * responsibility of the caller.
    *
    * @details If the window owns an allocated memory buffer, it will be automatically freed.
    *          Otherwise, only the MPI window handle is released.
    */
    virtual ~window() { free(); }

    explicit operator MPI_Win() const noexcept { return _win; };
    explicit operator MPI_Win *() noexcept { return &_win; };

    void free() noexcept {
      if (has_env) {
        if (_win != MPI_WIN_NULL) {
          this->fence();
          MPI_Win_free(&_win);
        }
      } else {
        if (_owned) { delete[] _data; }
        _data = nullptr;
        _size = 0;
      }
    }

    /**
    * @brief Synchronizes all RMA operations within an access epoch.
    *
    * @details This function acts as a barrier for remote memory access (RMA)
    *          operations, ensuring all previous operations on the window are
    *          completed before continuing.  The call is collective on the group
    *          of the window.
    *
    * @param assert program assertion.
    */
    void fence(int assert = 0) const noexcept {
      if (has_env) { MPI_Win_fence(assert, _win); }
    }

    /**
    * @brief Ensures completion of all outstanding RMA operations.
    *
    * @details This function forces all RMA operations issued to a specific rank (or all ranks)
    *          to complete at both the origin and the target before proceeding.
    *
    * @param rank The rank to flush operations for. If negative or no rank is specified, flushes all ranks .
    */
    void flush(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_flush_all(_win);
        } else {
          MPI_Win_flush(rank, _win);
        }
      }
    }

    /**
    * @brief Synchronizes the public and private copies of the window.
    *
    * @details Ensures that any updates to the local memory are visible in the public window
    *          and vice versa.
    */
    void sync() const noexcept {
      if (has_env) { MPI_Win_sync(_win); }
    }

    /**
    * @brief Starts an RMA access epoch.
    *
    * @details Locks access to the memory window for a specific rank or all ranks,
    *          preventing concurrent modifications.
    *
    * @param rank The rank to lock access for.
    * @param lock_type The type of lock (e.g., @p MPI_LOCK_SHARED or @p MPI_LOCK_EXCLUSIVE).
    * @param assert An assertion flag providing optimization hints to MPI.
    */
    void lock(int rank = -1, int lock_type = MPI_LOCK_SHARED, int assert = 0) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_lock_all(assert, _win);
        } else {
          MPI_Win_lock(lock_type, rank, assert, _win);
        }
      }
    }

    /**
    * @brief Completes an RMA access epoch started by @p lock().
    *
    * @details Unlocks access to the memory window for a specific rank or all ranks,
    *          allowing other processes to access or modify the window.
    *
    * @see lock
    *
    * @param rank The rank to unlock access for.
    */
    void unlock(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_unlock_all(_win);
        } else {
          MPI_Win_unlock(rank, _win);
        }
      }
    }

    /**
    * @brief Starts an RMA access epoch.
    *
    * @param grp The group of target processes.
    * @param assert An assertion flag providing optimization hints to MPI.
    */
    void start(group const &grp, int assert = 0) const noexcept {
      if (has_env) { MPI_Win_start(grp.get(), assert, _win); }
    }

    /**
    * @brief Completes an RMA access epoch on win started by a call to @p start.
    */
    void complete() const noexcept {
      if (has_env) { MPI_Win_complete(_win); }
    }

    /**
    * @brief Starts an RMA exposure epoch for the local window.
    *
    * @param grp The group of origin processes.
    * @param assert An assertion flag providing optimization hints to MPI.
    */
    void post(group const &grp, int assert = 0) const noexcept {
      if (has_env) { MPI_Win_post(grp.get(), assert, _win); }
    }

    /**
    * @brief Completes an RMA exposure epoch started by a call to @p post.
    */
    void wait() const noexcept {
      if (has_env) { MPI_Win_wait(_win); }
    }

    /**
    * @brief Reads data from a remote memory window.
    *
    * @details This function retrieves data from a remote process's memory
    *          window and stores it in a local buffer.
    *
    * @tparam TargetType The data type at the target memory.
    * @tparam OriginType The data type at the origin memory.
    * @param origin_addr Pointer to the memory buffer where the data will be stored.
    * @param origin_count Number of elements to retrieve.
    * @param target_rank Rank of the target process from which data is fetched.
    * @param target_disp Displacement (in @p disp_unit) from the start of the target memory window.
    * @param target_count Number of elements to read from the target. If negative or not specified, defaults to @p origin_count.
    */
    template <typename TargetType = BaseType, typename OriginType>
      requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void get(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      int target_count_ = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        MPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, _win);
      } else {
        if (target_rank != 0) { return; }

        std::span<OriginType> origin(origin_addr, origin_count);
        BaseType *target_begin = _data;
        std::advance(target_begin, target_disp);
        BaseType *target_end = target_begin;
        std::advance(target_end, target_count_);
        std::copy(target_begin, target_end, origin.begin());
      }
    }

    /**
    * @brief Writes data to a remote memory window.
    *
    * @details This function transfers data from a local buffer to a remote process's
    *          memory window.
    *
    * @tparam TargetType The data type at the target memory.
    * @tparam OriginType The data type at the origin memory.
    * @param origin_addr Pointer to the local memory buffer containing the data to be sent.
    * @param origin_count Number of elements to transfer.
    * @param target_rank Rank of the target process to which data is written.
    * @param target_disp Displacement (in @p disp_unit) from the start of the target memory window.
    * @param target_count Number of elements to write to the target. If negative or not specified, defaults to @p origin_count.
    */
    template <typename TargetType = BaseType, typename OriginType>
      requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void put(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      int target_count_ = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        MPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, _win);
      } else {
        if (target_rank != 0) { return; }

        std::span<OriginType> origin(origin_addr, origin_count);
        BaseType *target_begin = _data;
        std::advance(target_begin, target_disp);
        std::copy(origin.begin(), origin.end(), target_begin);
      }
    }

    /**
    * @brief Retrieves the value of a window attribute.
    *
    * @details This function queries an attribute associated with an MPI window.
    *
    * @param win_keyval The key identifying the attribute.
    * @return A pointer to the attribute value.
    */
    void *get_attr(int win_keyval) const noexcept {
      if (has_env) {
        int flag;
        void *attribute_val;
        MPI_Win_get_attr(_win, win_keyval, &attribute_val, &flag);
        ASSERT(flag)
        return attribute_val;
      } else {
        ASSERT(has_env)
        return nullptr;
      }
    }

    /**
    * @brief Retrieves the base address of the memory window.
    *
    * @details This function returns a pointer to the base address of the memory associated with the MPI window.
    *
    * @return A pointer to the base address of the window memory.
    */
    BaseType *base() const noexcept {
      if (has_env) {
        if (_win == MPI_WIN_NULL) { return nullptr; }
        return static_cast<BaseType *>(get_attr(MPI_WIN_BASE));
      } else {
        return _data;
      }
    }

    /**
    * @brief Retrieves the size of the memory window.
    *
    * @details This function returns the total size (in bytes) of the memory associated with the MPI window.
    *
    * @return The size of the MPI window in bytes.
    */

    MPI_Aint size() const noexcept {
      if (has_env) {
        return *static_cast<MPI_Aint *>(get_attr(MPI_WIN_SIZE));
      } else {
        return _size * sizeof(BaseType);
      }
    }

    /**
    * @brief Retrieves the displacement unit of the memory window.
    *
    * @details The displacement unit determines the scaling factor for address displacements.
    *
    * @return The displacement unit (in bytes).
    */

    int disp_unit() const noexcept {
      if (has_env) {
        return *static_cast<int *>(get_attr(MPI_WIN_DISP_UNIT));
      } else {
        return sizeof(BaseType);
      }
    }

    BaseType *&data() noexcept { return _data; }
    BaseType &data() const noexcept { return _data; }

    communicator get_communicator() noexcept { return _comm.get(); }
  };

  /**
  * @brief A shared memory window abstraction using MPI.
  *
  * @details This class provides an interface for creating and managing an MPI shared memory window.
  *
  * @tparam BaseType The data type stored in the shared memory window.
  */
  template <class BaseType> class shared_window : public window<BaseType> {
    public:
    /// Default constructor
    shared_window() = default;

    /**
     * @brief Constructs a shared memory window.
     *
     * @details This constructor allocates shared memory within the given communicator.
     *
     * @param c The shared communicator.
     * @param size The number of elements of type @p BaseType to allocate.
     * @param info MPI_Info object for optimization hints.
     */
    explicit shared_window(shared_communicator const &c, MPI_Aint size, MPI_Info info = MPI_INFO_NULL) noexcept {
      ASSERT(size >= 0)
      if (has_env) {
        void *baseptr = nullptr;
        MPI_Win_allocate_shared(size * sizeof(BaseType), sizeof(BaseType), info, c.get(), &baseptr, &(this->_win));
        this->_comm = c.get();
        this->_data = static_cast<BaseType *>(baseptr);
        this->_size = size;
      } else {
        this->_owned = true;
        this->_comm  = c.get();
        this->_data  = new BaseType[size];
        this->_size  = size;
      }
    }

    /**
     * @brief Queries attributes of a shared memory window.
     *
     * @details Retrieves the size, displacement unit, and base address of the shared memory region for a given rank.
     *
     * @param rank The rank within the communicator (defaults to @p MPI_PROC_NULL for querying all ranks).
     * @return A tuple containing (size in bytes, displacement unit, base pointer).
     */
    std::tuple<MPI_Aint, int, void *> query(int rank = MPI_PROC_NULL) const noexcept {
      if (has_env) {
        MPI_Aint size = 0;
        int disp_unit = 0;
        void *baseptr = nullptr;
        MPI_Win_shared_query(this->_win, rank, &size, &disp_unit, &baseptr);
        return {size, disp_unit, baseptr};
      } else {
        return {this->_size * sizeof(BaseType), sizeof(BaseType), this->_data};
      }
    }

    // Override the commonly used attributes of the window base class

    /**
     * @brief Returns the base address of the shared memory for a specific rank.
     *
     * @param rank The rank whose base address should be retrieved.
     * @return A pointer to the base address.
     */
    BaseType *base(int rank = MPI_PROC_NULL) const noexcept { return static_cast<BaseType *>(std::get<2>(query(rank))); }

    /**
     * @brief Returns the number of elements stored in the shared memory window.
     *
     * @param rank The rank whose memory size should be retrieved.
     * @return The number of elements in the shared window.
     */
    MPI_Aint size(int rank = MPI_PROC_NULL) const noexcept { return std::get<0>(query(rank)) / sizeof(BaseType); }

    /**
     * @brief Returns the displacement unit of the shared memory.
     *
     * @param rank The rank whose displacement unit should be retrieved.
     * @return The displacement unit.
     */
    int disp_unit(int rank = MPI_PROC_NULL) const noexcept { return std::get<1>(query(rank)); }

    shared_communicator get_communicator() { return this->_comm.get(); }
  };

  /** @} */

} // namespace mpi
