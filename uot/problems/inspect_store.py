import argparse
import matplotlib.pyplot as plt

from uot.problems.store import ProblemStore
from uot.problems.iterator import ProblemIterator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect storage file.")
    parser.add_argument(
        "--store",
        type=str,
        required=True,
        help="Path to the storage file to inspect."
    )
    args = parser.parse_args()

    store = ProblemStore(args.store)
    iterator = ProblemIterator(store)

    for problem in iterator:
        mu, nu = problem.get_marginals()

        mu_points, mu_weights = mu.to_discrete()
        nu_points, nu_weights = nu.to_discrete()

        plt.plot(mu_points, mu_weights)
        plt.plot(nu_points, nu_weights)

        plt.show()



