#include <mpi/mpi.hpp>
#include <iostream>
#include <vector>

// Custom type.
class foo {
  public:
  // Constructor.
  foo(int x = 5) : x_(x) {}

  // Get the value stored in the class.
  [[nodiscard]] int x() const { return x_; }

  // Specialization of mpi_reduce_into for the custom type.
  friend void mpi_reduce_into(foo const &f_in, foo &f_out, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    mpi::reduce_into(f_in.x_, f_out.x_, c, root, all, op);
  }

  private:
  int x_;
};

int main(int argc, char *argv[]) {
  // initialize MPI environment
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // create a vector of foo objects
  std::vector<foo> vec{foo{1}, foo{2}, foo{3}, foo{4}, foo{5}};

  // reduce the vector of foo objects
  auto result = mpi::reduce(vec, world);

  // print the result on rank 0
  if (world.rank() == 0) {
    std::cout << "Reduced vector: ";
    for (auto const &f : result) std::cout << f.x() << " ";
    std::cout << "\n";
  }
}