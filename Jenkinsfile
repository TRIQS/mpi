def projectName = "mpi" /* set to app/repo name */

def dockerName = projectName.toLowerCase();
/* which platform to build documentation on */
def documentationPlatform = "ubuntu-clang"
/* whether to keep and publish the results */
def publish = !env.BRANCH_NAME.startsWith("PR-")

properties([
  buildDiscarder(logRotator(numToKeepStr: '10', daysToKeepStr: '30'))
])
