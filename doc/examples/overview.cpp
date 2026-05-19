#include <mpi/mpi.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  // initialize MPI environment and communicator
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // get rank of process
  int rank = world.rank();

  // perform a reduce operation
  int sum = mpi::reduce(rank, world);

  // output the result on root
  if (rank == 0) {
    std::cout << "The sum of all ranks is " << sum << "\n";
  }
}
