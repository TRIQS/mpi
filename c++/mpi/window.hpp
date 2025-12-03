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
 * @brief Provides a C++ wrapper class for an `MPI_Win` object.
 */

#pragma once

#include "./communicator.hpp"
#include "./datatypes.hpp"
#include "./group.hpp"
#include "./macros.hpp"
#include "./utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>

namespace mpi {

  /**
   * @addtogroup mpi_osc_shm
   * @{
   */

  /**
   * @brief A C++ wrapper around `MPI_Win` providing convenient memory window management.
   *
   * @details This class abstracts the complexities of MPI window management, allowing processes in an MPI communicator
   * to create and share memory regions efficiently. It supports both local buffer-based windows and dynamically
   * allocated memory windows.
   *
   * If a base pointer is not specified, the constructor will allocate memory internally.
   *
   * This class follows move-only semantics and takes ownership of the wrapped `MPI_Win` object.
   * 
   * All functions that make direct calls to the MPI C API, except free(), check their success with mpi::check_mpi_call.
   *
   * @tparam BaseType The type of elements stored in the memory window.
   */
  template <class BaseType> class window {
    public:
    /// Type of the base pointer.
    using base_type = BaseType;

    /// Construct a window with `MPI_WIN_NULL`.
    window() = default;

    /// Deleted copy constructor.
    window(window const &) = delete;

    /// Deleted copy assignment operator.
    window &operator=(window const &) = delete;

    /// Move constructor takes ownership of the moved-from MPI window and leaves it with `MPI_WIN_NULL`.
    window(window &&other) noexcept
       : win_{std::exchange(other.win_, MPI_WIN_NULL)},
         comm_{std::exchange(other.comm_, communicator{MPI_COMM_NULL})},
         owned_{std::exchange(other.owned_, false)},
         data_{std::exchange(other.data_, nullptr)},
         size_{std::exchange(other.size_, 0)} {}

    /// Move assignment operator takes ownership of the moved-from MPI window and leaves it with `MPI_WIN_NULL`.
    window &operator=(window &&rhs) noexcept {
      if (this != std::addressof(rhs)) {
        free();
        win_   = std::exchange(rhs.win_, MPI_WIN_NULL);
        comm_  = std::exchange(rhs.comm_, communicator{MPI_COMM_NULL});
        owned_ = std::exchange(rhs.owned_, false);
        data_  = std::exchange(rhs.data_, nullptr);
        size_  = std::exchange(rhs.size_, 0);
      }
      return *this;
    }

    /**
     * @brief Construct an MPI window over an existing local memory buffer.
     *
     * @details This constructor allows creating a window using a pre-allocated memory buffer by calling
     * `MPI_Win_create`. The window provides access to the specified memory region across MPI processes within the given
     * communicator. The buffer is not freed upon destruction.
     *
     * @param c mpi::communicator that defines the group of processes sharing the window.
     * @param base_ptr Pointer to the base address of the memory buffer.
     * @param sz Number of elements in the buffer.
     * @param info Additional MPI information. Default is `MPI_INFO_NULL`.
     */
    explicit window(communicator const &c, BaseType *base_ptr, MPI_Aint sz, MPI_Info info = MPI_INFO_NULL)
       : comm_(c.get()), data_(base_ptr), size_(sz) {
      ASSERT(size_ >= 0)
      ASSERT(!(data_ == nullptr && size_ > 0))
      if (has_env) check_mpi_call(MPI_Win_create(data_, size_ * sizeof(BaseType), sizeof(BaseType), info, c.get(), &win_), "MPI_Win_create");
    }

    /**
     * @brief Construct an MPI window with dynamically allocated memory.
     *
     * @details This constructor allocates a new memory buffer locally and creates an MPI window over it by calling
     * `MPI_Win_allocate`. The allocated memory is automatically freed when the window is destroyed. This is useful when
     * the memory region is meant to be shared across processes without needing an external buffer.
     *
     * @param c mpi::communicator that defines the group of processes sharing the window.
     * @param sz Number of elements to allocate for the calling process.
     * @param info Additional MPI information. Default is `MPI_INFO_NULL`.
     */
    explicit window(communicator const &c, MPI_Aint sz, MPI_Info info = MPI_INFO_NULL) : comm_(c.get()), size_(sz) {
      ASSERT(size_ >= 0)
      if (has_env) {
        check_mpi_call(MPI_Win_allocate(size_ * sizeof(BaseType), sizeof(BaseType), info, c.get(), &data_, &win_), "MPI_Win_allocate");
      } else {
        data_ = new BaseType[size_]; // NOLINT (new is fine here)
      }
      owned_ = true;
    }

    /// Convert the window to the wrapped `MPI_Win` object.
    explicit operator MPI_Win() const { return win_; };

    /// Convert a pointer to the window to a pointer to the wrapped `MPI_Win` object.
    explicit operator MPI_Win *() { return &win_; };

    /// Destructor calls free() to release the window.
    virtual ~window() { free(); }

    /**
     * @brief Release allocated resources owned by the window.
     *
     * @details Before freeing the owned memory or the `MPI_Win` handle, a window must have completed all its
     * involvement in RMA communications. For that reason we call `MPI_Win_fence` before `MPI_Win_free`.
     *
     * The window also must be unlocked if it has been previously locked. However, this cannot be detected and is
     * therefore the responsibility of the user.
     *
     * If the window owns an allocated memory buffer, it will be automatically freed. Otherwise, only the MPI window
     * handle is released.
     */
    void free() noexcept {
      if (has_env) {
        if (win_ != MPI_WIN_NULL) {
          MPI_Win_fence(0, win_);
          MPI_Win_free(&win_);
        }
      } else if (owned_) {
        delete[] data_;
      }
      owned_ = false;
      data_  = nullptr;
      size_  = 0;
    }

    /**
     * @brief Synchronize all RMA operations within an access epoch by calling `MPI_Win_fence`.
     *
     * @details This function acts as a barrier for remote memory access (RMA) operations, ensuring all previous
     * operations on the window are completed before continuing. The call is collective on the group of the window.
     *
     * @param assert Program assertion.
     */
    void fence(int assert = 0) const {
      if (has_env) check_mpi_call(MPI_Win_fence(assert, win_), "MPI_Win_fence");
    }

    /**
     * @brief Ensure completion of all outstanding RMA operations.
     *
     * @details If the given target rank is \f$ < 0 \f$, it calls `MPI_Win_flush_all`. Otherwise, it calls
     * `MPI_Win_flush`.
     *
     * @param rank Target rank.
     */
    void flush(int rank = -1) const {
      if (has_env) {
        if (rank < 0) {
          check_mpi_call(MPI_Win_flush_all(win_), "MPI_Win_flush_all");
        } else {
          check_mpi_call(MPI_Win_flush(rank, win_), "MPI_Win_flush");
        }
      }
    }

    /**
     * @brief Synchronize the public and private copies of the window.
     *
     * @details It ensures that any updates to the local memory are visible in the public window and vice versa by
     * calling `MPI_Win_sync`.
     */
    void sync() const {
      if (has_env) check_mpi_call(MPI_Win_sync(win_), "MPI_Win_sync");
    }

    /**
     * @brief Start an RMA access epoch.
     *
     * @details It locks access to the memory window on a specific rank or all ranks, preventing concurrent
     * modifications.
     *
     * If the given target rank is \f$ < 0 \f$, it calls `MPI_Win_lock_all`. Otherwise, it calls `MPI_Win_lock`.
     *
     * @param rank Target rank.
     * @param lock_type Type of the lock (e.g. `MPI_LOCK_SHARED` or `MPI_LOCK_EXCLUSIVE`).
     * @param assert An assertion flag providing optimization hints to MPI.
     */
    void lock(int rank = -1, int lock_type = MPI_LOCK_SHARED, int assert = 0) const {
      if (has_env) {
        if (rank < 0) {
          check_mpi_call(MPI_Win_lock_all(assert, win_), "MPI_Win_lock_all");
        } else {
          check_mpi_call(MPI_Win_lock(lock_type, rank, assert, win_), "MPI_Win_lock");
        }
      }
    }

    /**
     * @brief Complete an RMA access epoch started by lock().
     *
     * @details It unlocks access to the memory window on a specific rank or all ranks, allowing other processes to
     * access or modify the window.
     *
     * If the given target rank is \f$ < 0 \f$, it calls `MPI_Win_unlock_all`. Otherwise, it calls `MPI_Win_unlock`.
     *
     * @param rank Target rank.
     */
    void unlock(int rank = -1) const {
      if (has_env) {
        if (rank < 0) {
          check_mpi_call(MPI_Win_unlock_all(win_), "MPI_Win_unlock_all");
        } else {
          check_mpi_call(MPI_Win_unlock(rank, win_), "MPI_Win_unlock");
        }
      }
    }

    /**
     * @brief Start an RMA access epoch by calling `MPI_Win_start` (see also complete()).
     *
     * @param grp mpi::group of target processes.
     * @param assert An assertion flag providing optimization hints to MPI.
     */
    void start(group const &grp, int assert = 0) const {
      if (has_env) check_mpi_call(MPI_Win_start(grp.get(), assert, win_), "MPI_Win_start");
    }

    /// Completes an RMA access epoch by calling `MPI_Win_complete` (see also start()).
    void complete() const {
      if (has_env) check_mpi_call(MPI_Win_complete(win_), "MPI_Win_complete");
    }

    /**
     * @brief Start an RMA exposure epoch by calling `MPI_Win_post` (see also wait()).
     *
     * @param grp mpi::group of origin processes.
     * @param assert An assertion flag providing optimization hints to MPI.
     */
    void post(group const &grp, int assert = 0) const {
      if (has_env) check_mpi_call(MPI_Win_post(grp.get(), assert, win_), "MPI_Win_post");
    }

    /// Completes an RMA exposure epoch by calling `MPI_Win_wait` (see also post()).
    void wait() const {
      if (has_env) check_mpi_call(MPI_Win_wait(win_), "MPI_Win_wait");
    }

    /**
     * @brief Read data from a remote memory window.
     *
     * @details This function retrieves data from the memory window on the given process by calling `MPI_get` and stores
     * it in a local buffer.
     *
     * @tparam TargetType Value type of the target memory.
     * @tparam OriginType Value type of the origin memory.
     * @param origin_addr Pointer to the memory buffer where the data will be stored.
     * @param origin_count Number of elements to retrieve.
     * @param target_rank Rank of the target process from which data is fetched.
     * @param target_disp Displacement from the start of the target memory window.
     * @param target_count Number of elements to read from the target. If negative or not specified, defaults to
     * `origin_count`.
     */
    template <typename TargetType = BaseType, typename OriginType>
      requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void get(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const {
      ASSERT(origin_count >= 0 && target_disp >= 0);
      target_count = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        auto origin_datatype = mpi_type<OriginType>::get();
        auto target_datatype = mpi_type<TargetType>::get();
        check_mpi_call(MPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win_), "MPI_Get");
      } else {
        ASSERT(target_rank == 0); // there are no other ranks
        std::copy(data_, data_ + target_count, origin_addr);
      }
    }

    /**
     * @brief Write data to a remote memory window.
     *
     * @details This function transfers data from a local buffer to the memory window on the given process by calling
     * `MPI_Put`.
     *
     * @tparam TargetType Value type at the target memory.
     * @tparam OriginType Value type at the origin memory.
     * @param origin_addr Pointer to the local memory buffer containing the data to be sent.
     * @param origin_count Number of elements to transfer.
     * @param target_rank Rank of the target process to which data is written.
     * @param target_disp Displacement from the start of the target memory window.
     * @param target_count Number of elements to write to the target. If negative or not specified, defaults to
     * `origin_count`.
     */
    template <typename TargetType = BaseType, typename OriginType>
      requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void put(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const {
      ASSERT(origin_count >= 0 && target_disp >= 0);
      target_count = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        auto origin_datatype = mpi_type<OriginType>::get();
        auto target_datatype = mpi_type<TargetType>::get();
        check_mpi_call(MPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win_), "MPI_Put");
      } else {
        ASSERT(target_rank == 0); // there are no other ranks
        std::copy(origin_addr, origin_addr + origin_count, data_);
      }
    }

    /// Get a pointer to the beginning of the window memory.
    [[nodiscard]] BaseType *base() const { return data_; }

    /// Get the size of the window in number of elements.
    [[nodiscard]] MPI_Aint size() const { return size_; }

    /// Get the displacement unit in bytes.
    [[nodiscard]] int disp_unit() const { return sizeof(BaseType); }

    /// Get the mpi::communicator associated with the window.
    [[nodiscard]] communicator get_communicator() const { return comm_; }

    protected:
    MPI_Win win_{MPI_WIN_NULL};
    communicator comm_{MPI_COMM_NULL};
    bool owned_{false};
    BaseType *data_{nullptr};
    MPI_Aint size_{0};
  };

  /**
   * @brief A C++ wrapper around `MPI_Win` representing a shared memory window.
   *
   * @details This class provides an interface for creating and managing an MPI shared memory window.
   * 
   * All functions that make direct calls to the MPI C API check their success with mpi::check_mpi_call.
   *
   * @tparam BaseType The type of elements stored in the shared memory window.
   */
  template <class BaseType> class shared_window : public window<BaseType> {
    public:
    /// Construct a shared memory window with `MPI_WIN_NULL`.
    shared_window() = default;

    /**
     * @brief Construct a shared memory window by dynamically allocating memory.
     *
     * @details This constructor allocates a shared memory window within the given communicator by calling
     * `MPI_Win_allocate_shared`. The allocated memory is automatically freed when the window is destroyed.
     *
     * @param c mpi::shared_communicator object.
     * @param sz Number of elements to allocate for the calling process.
     * @param info Additional MPI information.
     */
    explicit shared_window(shared_communicator const &c, MPI_Aint sz, MPI_Info info = MPI_INFO_NULL) {
      ASSERT(sz >= 0)
      comm_ = c.get();
      size_ = sz;
      if (has_env) {
        check_mpi_call(MPI_Win_allocate_shared(size_ * sizeof(BaseType), sizeof(BaseType), info, c.get(), &data_, &win_), "MPI_Win_allocate_shared");
      } else {
        data_ = new BaseType[size_]; // NOLINT (new is fine here)
      }
      owned_ = true;
    }

    /**
     * @brief Query attributes of a shared memory window.
     *
     * @details Retrieves the byte-size, displacement unit, and a pointer to the beginning of the shared memory region 
     * for a specific rank.
     *
     * When `MPI_PROC_NULL` is passed for the rank, MPI returns information about the memory segment with the lowest 
     * rank that has a non-zero size.
     *
     * @param rank Rank within the shared communicator.
     * @return A tuple containing the byte-size, the displacement unit in bytes and the base pointer.
     */
    [[nodiscard]] std::tuple<MPI_Aint, int, void *> query(int rank = MPI_PROC_NULL) const {
      if (has_env) {
        MPI_Aint sz   = 0;
        int du        = 0;
        void *baseptr = nullptr;
        check_mpi_call(MPI_Win_shared_query(win_, rank, &sz, &du, &baseptr), "MPI_Win_shared_query");
        return {sz, du, baseptr};
      } else {
        return {size_, sizeof(BaseType), data_};
      }
    }

    /**
     * @brief Get a pointer to the beginning of the shared memory region of a specific rank.
     *
     * @param rank Rank within the shared communicator.
     * @return Pointer to the shared window of the given rank.
     */
    [[nodiscard]] BaseType *base(int rank = MPI_PROC_NULL) const { return static_cast<BaseType *>(std::get<2>(query(rank))); }

    /**
     * @brief Get the size of the shared memory region of a specific rank.
     *
     * @param rank Rank within the shared communicator.
     * @return Number of elements in the shared window of the given rank.
     */
    [[nodiscard]] MPI_Aint size(int rank = MPI_PROC_NULL) const { return std::get<0>(query(rank)) / sizeof(BaseType); }

    /**
     * @brief Get the displacement unit of the shared memory region of a specific rank.
     *
     * @param rank Rank within the shared communicator.
     * @return Displacement unit in bytes.
     */
    [[nodiscard]] int disp_unit(int rank = MPI_PROC_NULL) const { return std::get<1>(query(rank)); }

    /// Get the mpi::shared_communicator associated with the window.
    [[nodiscard]] shared_communicator get_communicator() const { return comm_.get(); }

    private:
    using window<BaseType>::win_;
    using window<BaseType>::comm_;
    using window<BaseType>::owned_;
    using window<BaseType>::data_;
    using window<BaseType>::size_;
  };

  /** @} */

} // namespace mpi
