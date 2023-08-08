// Copyright (c) 2020-2022 Simons Foundation
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
// Authors: Philipp Dumitrescu, Olivier Parcollet, Nils Wentzell

#pragma once
#include "mpi/mpi.hpp"
#include "mpi/macros.hpp"
#include <vector>
#include <algorithm>
#include <unistd.h>

namespace mpi {

  /**
   * Contructed on top on a mpi communicator, this class
   * monitors and syncronizes failure states of nodes, due to e.g. exceptions.
   *
   * Usage : 
   *   monitor M{comm};
   *
   *   on a node when failure occurs : request_emergency_stop.
   *
   *   on all nodes : emergency_occured -> bool tells where a node a requested an emergency stop
   *
   *   finalize_communications() cleans the monitor object and returns a bool : true iif computation finished normally (no emergency stop).
   */
  class monitor {

    mpi::communicator com;

    struct future {
      MPI_Request request{};
      int value = 0;
    };
    std::vector<future> root_futures;      // communication of local_stop from the nodes to the root. On root only.
    MPI_Request req_ibcast{}, req_isent{}; // request for the ibcast and isent. On all nodes.

    int local_stop  = 0;     // = 1 if the node has requested an emergency stop. Local to the node. (No bool in MPI.)
    int global_stop = 0;     // = 1 if any node has requested an emergency stop. Always the same on all nodes.
    bool finalized  = false; // the finalized() method should be called once,
                             // and can not just be the desctructor since we use the returned value

    public:
    /// Constructs on top on a mpi communicator
    monitor(mpi::communicator c) : com(c) {
      if (com.rank() == 0) { // on root
        // Register an async recv for the variable local_stop from every non-root nodes
        // The associated send will be issued from each node:
        // * with value = 1 in case of emergency stop
        // * with value = 0 during finalization
        root_futures.resize(c.size() - 1);
        for (int rank = 1; rank < c.size(); ++rank) { // the index of root_futures is rank - 1 since there is none for rank = 0
          MPI_Irecv(&(root_futures[rank - 1].value), 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &(root_futures[rank - 1].request));
        }
      } else { // not root
        // Register the receive of the ibcast of global_stop that root will issue (in case of emergency stop or during fianlize)
        MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast);
      }
    }

    monitor(monitor const &)            = delete;
    monitor &operator=(monitor const &) = delete;

    ~monitor() { finalize_communications(); }

    /// Request an emergency stop of all nodes contained in the mpi communicator.
    /// It send the message to the root.
    /// It the local node is the root, immedeatly sends the global ibcast to all nodes can check if to stop
    void request_emergency_stop() {
      EXPECTS(!finalized);
      if (local_stop) { return; } // prevent sending signal multiple times
      local_stop = 1;
      if (com.rank() == 0) { // root
        global_stop = 1;
        MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast);
      } else { // non root
        MPI_Isend(&local_stop, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_isent);
      }
    }

    /// Any node: should the calculation should be stopped?
    [[nodiscard]] bool emergency_occured() {
      if (finalized) return global_stop;
      if (global_stop or local_stop) return true;
      if (com.rank() == 0) { // root test whether other nodes have emergency stop and communicates signal
        root_check_nodes_and_bcast();
      } else { // other nodes just listen to the root bcast to see if an emergency stop broadcast has occured
        MPI_Status status;
        int flag = 0;
        MPI_Test(&req_ibcast, &flag, &status);
        // if flag, then global_stop is now what was bcasted by the root
      }
      return global_stop;
    }

    /// Finalize the monitor.
    /// At end of this functions, all nodes have completed their work, or have had an emergency stop.
    /// The global_stop result is guaranteed to be the same on all nodes.
    void finalize_communications() {
      if (finalized) return;
      if (com.rank() == 0) {
        // the root is done computing, it just listens to the other nodes and bcast the global_stop until everyone is done.
        while (root_check_nodes_and_bcast()) { usleep(100); } // 100 us (micro seconds)
        // all others node have finished
        // if the root has never emitted the ibcast, we do it and wait it, since we can not cancel it FIXME (why ??).
        if (!global_stop) { MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast); }
      } else {
        // on non-root node: either Isend was done when local_stop was set (during request_emergency_stop),
        // or it has to happen now, sending local_stop = 0, i.e, work is done, and fine.
        if (!local_stop) { MPI_Isend(&local_stop, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_isent); }
      }
      // All nodes wait for the ibcast of global_stop to be complete.
      MPI_Status status;
      MPI_Wait(&req_ibcast, &status);
      finalized = true;
    }

    private:
    // ROOT ONLY.
    // looks at all irecv that have arrived, and if they are a stop request, bcast the global stop.
    // returns true iff there are still running nodes.
    bool root_check_nodes_and_bcast() {
      EXPECTS(!finalized);
      EXPECTS(com.rank() == 0); // root only
      bool some_nodes_are_still_running = false;
      for (auto &f : root_futures) {
        MPI_Status status;
        int flag = 0;
        MPI_Test(&(f.request), &flag, &status);
        if (flag and (not global_stop) and (f.value > 0)) request_emergency_stop(); // the root requires the stop now. It also stops itself...
        some_nodes_are_still_running |= (flag == 0);
      }
      return some_nodes_are_still_running;
    }
  };

} // namespace mpi
