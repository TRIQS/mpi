#pragma once
#include "mpi/mpi.hpp"
#include "mpi/macros.hpp"
#include <vector>
#include <algorithm>
#include <unistd.h>

namespace mpi {

  /**
   * Contructed on top on a communicator, this class
   * monitors the failure of node, due to e.g. exceptions.
   *
   * Usage : 
   *   monitor M{comm};
   *
   *   on a node : request_emergency_stop.
   *
   *   on all nodes : should_stop -> bool tells where a node a requested an emergency stop
   *
   *   At the end, finalize() clean the monitor object and return a bool : true iif it finished normally. 
   */
  class monitor {

    mpi::communicator com;

    struct future {
      MPI_Request request;
      int value = 0;
    };
    std::vector<future> root_futures;  // the communication of local_stop from the nodes to the root. On the root only.
    MPI_Request req_ibcast, req_isent; // request for the ibcast and isent. all nodes.

    int local_stop  = 0;     // 1 if the node has requested an emergency stop. Local to the node. No bool in MPI.
    int global_stop = 0;     // 1 if one node has requested an emergency stop. Always the same on all nodes.
    bool finalized  = false; // the finalized should be called once,
                             // and can not be the desctructor since we use the returned value

    public:
    /// Constructs on top on a communicator
    monitor(mpi::communicator c) : com(c) {
      if (com.rank() == 0) { // on root
        root_futures.resize(c.size() - 1);
        // place a async recv for local_stop from all OTHER nodes
        // the send will be issued in case of emergency stop with value 1, or at the end with value 0
        for (int rank = 1; rank < c.size(); ++rank) { // the index of root_futures is rank - 1 since there is none for rank = 0
          MPI_Irecv(&(root_futures[rank - 1].value), 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &(root_futures[rank - 1].request));
        }
      } else { // not root
        // place the receive of the ibcast of global_stop that root will issue in case of emergency stop
        MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast);
      }
    }

    monitor(monitor const &) = delete;
    monitor &operator=(monitor const &) = delete;

    ~monitor() { finalize(); }

    // should only called at the end. After calling it, you can not call should_stop or request_emergency_stop
    // Guarantees the same answer on all nodes
    [[nodiscard]] bool success() {
      finalize();
      return not global_stop;
    }

    /// Request an emergency stop of all nodes in the mpi Communicator.
    /// It send the message to the root.
    /// It the local node is the root, it sends the global bcast to all nodes to order them to stop
    void request_emergency_stop() {
      EXPECTS(!finalized);
      local_stop  = 1;
      global_stop = 1;
      if (com.rank() == 0) // root
        MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast);
      else // non root
        MPI_Isend(&local_stop, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_isent);
    }

    /// Any node : should I stop my calculation ?
    bool should_stop() {
      EXPECTS(!finalized);
      if (global_stop or local_stop) return true;
      if (com.rank() == 0) { // root test whether other nodes have emergency stop
        root_listen_and_bcast();
      } else { // other nodes just listen to the root bcast to see if an emergency stop has been broadcasted
        MPI_Status status;
        int flag;
        MPI_Test(&req_ibcast, &flag, &status);
        // if flag, then global_stop is now what was bcasted by the root
      }
      return global_stop;
    }

    private:
    // ROOT ONLY.
    // looks at all irecv that have arrived, and if they are a stop request, bcast the global stop.
    // returns true iff there are still running nodes.
    bool root_listen_and_bcast() {
      EXPECTS(!finalized);
      EXPECTS(com.rank() == 0); // root only
      bool some_nodes_are_still_running = false;
      for (auto &f : root_futures) {
        MPI_Status status;
        int flag;
        MPI_Test(&(f.request), &flag, &status);
        if (flag and (not global_stop) and (f.value > 0)) request_emergency_stop(); // the root requires the stop now. It also stops itself...
        some_nodes_are_still_running |= (flag == 0);
      }
      return some_nodes_are_still_running;
    }

    /// Finalize the monitor.
    /// As the end of this functions, all nodes have completed their work, or had an emergency stop.
    void finalize() {
      if (finalized) return;
      EXPECTS_WITH_MESSAGE(not finalized, "Logic error : mpi::monitor::finalize can only be called once.");
      if (com.rank() == 0) {
        // the root is done computing, it just listens to the other nodes and bcast the global_stop until everyone is done.
        while (root_listen_and_bcast()) {
        } //sleep(1); } // 1s is long for tests. To use a value in ms, I would need to include <chrono>, <thread>, cf tests.
        // all others node have finished
        // if the root has never emitted the ibcast, we do it and wait it, since we can not cancel it FIXME (why ??).
        if (!global_stop) MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast);
      } else {
        // on a node : either Isend was done when setting local_stop, or we do it now, sending local_stop = 0, i.e, work is done, and fine.
        if (!local_stop) MPI_Isend(&local_stop, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_isent);
      }
      // all nodes wait for the ibcast to be complete.
      MPI_Status status;
      MPI_Wait(&req_ibcast, &status);
      finalized = true;
    }
  };

} // namespace mpi
