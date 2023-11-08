from typing import Optional, Union

import numpy as np
from scipy import sparse

from src.data.representation.hypergraph import Hypergraph
from src.model._linear_ops import bf_and_sum
from src.model.HyMMSBM import HyMMSBM


class HyCoSBM(HyMMSBM):
    """Implements the HyCoSBM probabilistic model.

    HyCoSBM is a community detection model for hypergraphs with node attributes, see
    # TODO put reference here.
    """

    def __init__(
        self,
        K: Optional[int] = None,
        u: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: float = 0.0,
        assortative: Optional[bool] = None,
        kappa_fn: str = "binom+avg",
        max_hye_size: Optional[int] = None,
        u_prior: Union[float, np.ndarray] = 0.0,
        w_prior: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        K: Number of communities in the model.
        u: Soft community assignment matrix.
        If provided as input, it is considered fixed and will not be changed during
            inference. If provided it must be a matrix with non-negative entries and
            shape (N, K), with N the number of nodes and K the number of communities.
        w: Affinity matrix.
            If provided as input, it is considered fixed and will not be changed during
            inference. If provided it must be a matrix with non-negative entries and
            shape (K, K), with K the number of communities.
        beta: Matrix for prediction of the node attributes.
            If provided as input, it is considered fixed and will not be changed during
            inference. If provided it must be a matrix with non-negative entries and
            shape (K, Z), with K the number of communities and Z the number of classes
            of the attributes. The columns of the matrix need to sum to 1.
        gamma: Gamma parameter regulating the likelihood of the model.
        assortative: Whether the affinity matrix is diagonal or not.
        kappa_fn: The type of kappa normalization, by default "binom+avg".
        max_hye_size: Maximum hyperedge size consiered in the model.
        u_prior: Rate for the exponential prior on u.
            It can be provided as a float, in which case it specifies the same prior
            parameter for all the entries of u, or as an array of (possible different)
            separate exponential rates for every entry of u. If it is an array, it needs
            to have same shape as u.
            To avoid specifying a prior for u, set u_prior to 0.0.
        w_prior: Rate for the exponential prior on w.
            Similar to the exponential rate for u. If an array, it needs to be a
            symmetric matrix with same shape as w.
            To avoid specifying a prior for w, set w_prior to the 0. float value.
        seed: Random seed.
        """
        super().__init__(
            K, u, w, assortative, kappa_fn, max_hye_size, u_prior, w_prior, seed
        )

        # Scaling parameter
        self.gamma = gamma
        # Beta matrix
        self.beta = beta
        self._check_beta_consistency()

    def fit(
        self,
        hypergraph: Hypergraph,
        X: Optional[np.ndarray] = None,
        n_iter: int = 10,
        tolerance: Optional[float] = 0.0,
        check_convergence_every: int = 10,
    ) -> None:
        """Perform Expectation-Maximization inference on an attributed hypergraph, as
        presented  in

        TODO ADD PAPER

        The inference can be performed on the affinity matrix w, the community
        assignments u, and the beta matrix for attribute prediction.
        If any of these three parameters has been provided as input at initialization of
        the model, it is regarded as ground-truth and kept constant during inference.

        Parameters
        ----------
        hypergraph: the hypergraph to perform inference on.
        X: the array of attributes.
        n_iter: maximum number of EM iterations.
        tolerance: tolerance for the stopping criterion.
        check_convergence_every: number of steps in between every convergence check.
        """
        self._check_attributes_consistency(X)
        # Initialize all the values needed for training.
        self.tolerance = tolerance
        self.tolerance_reached = False

        if (self.w is None) and (self.gamma != 1):
            fixed_w = False
            self._init_w()
        elif self.w is None:
            self.w = np.eye(self.K)
            fixed_w = True
        else:
            fixed_w = True

        if self.u is None:
            fixed_u = False
            self._init_u(hypergraph)
        else:
            fixed_u = True

        if (self.beta is None) and (self.gamma != 0) and (X is not None):
            fixed_beta = False
            self._init_beta(X)
        elif self.beta is None:
            self.beta = np.zeros(1)
            fixed_beta = True
        else:
            fixed_beta = True

        # Infer the maximum hyperedge size if not already specified inside the model.
        if self.max_hye_size is None:
            self.max_hye_size = hypergraph.max_hye_size
        else:
            if self.max_hye_size < hypergraph.max_hye_size:
                raise ValueError(
                    "The hypergraph contains hyperedges with size greater than that "
                    "specified in the model. This will not influence training, but "
                    "might cause other modeling problems. If you want max_hye_size to "
                    "be detected automatically, set it to None."
                )

        binary_incidence = hypergraph.get_binary_incidence_matrix()
        hye_weights = hypergraph.get_hye_weights()

        # Train.
        for it in range(n_iter):
            old_w, old_u, old_beta = self.w, self.u, self.beta

            if not fixed_w:
                self.w = self._w_update(binary_incidence, hye_weights)
                self.w = np.clip(self.w, a_min=1e-17, a_max=None)
            if not fixed_u:
                new_u = self._u_update(binary_incidence, hye_weights, X)
                assert np.all(new_u >= -1e-16), new_u[~(new_u >= -1e-16)]
                self.u = np.clip(new_u, a_min=1e-17, a_max=1)
            if not fixed_beta:
                new_beta = self._beta_update(X)
                new_beta = np.clip(new_beta, a_min=1e-17, a_max=None)
                assert np.allclose(new_beta.sum(axis=0), 1), (
                    self.beta,
                    self.beta.shape,
                )
                self.beta = new_beta
            # Check for convergence.
            if tolerance is not None:
                if (not it % check_convergence_every) and (it > 0):
                    converged = (
                        np.linalg.norm(self.w - old_w) / self.K <= tolerance
                        and np.linalg.norm(self.u - old_u) / hypergraph.N <= tolerance
                        and np.linalg.norm(self.beta - old_beta) / self.K <= tolerance
                    )
                    if converged:
                        self.tolerance_reached = True
                        break

        self.trained = True
        self.training_iter = it

    def log_likelihood(
        self,
        hypergraph: Hypergraph,
        X: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the log-likelihood of the model on a given hypergraph.

        Parameters
        ----------
        hypergraph: the hypergraph to compute the log-likelihood of.

        Returns
        -------
        The log-likelihood value.
        """
        self._check_u_w_init()
        u, w, beta, gamma = self.u, self.w, self.beta, self.gamma

        if self.gamma != 1:
            binary_incidence = hypergraph.get_binary_incidence_matrix()
            hye_weights = hypergraph.get_hye_weights()

            # First addend: all interactions u_i * w * u_j .
            first_addend = bf_and_sum(u, w)

            # Second addend: interactions in the hypergraph A_e * log(lambda_e) .
            second_addend = np.dot(
                hye_weights, np.log(self.poisson_params(binary_incidence))
            )
            L_A = -self.C() * first_addend + second_addend
        else:
            L_A = 0

        if (X is None) or (self.gamma == 0):
            L_X = 0
        else:
            L_X = (X * np.log(u @ beta)).sum()

        return (1 - gamma) * L_A + gamma * L_X

    def _u_update(
        self,
        binary_incidence: Union[np.ndarray, sparse.spmatrix],
        hye_weights: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """EM updates for the community assignments u."""
        u, w, beta, gamma = self.u, self.w, self.beta, self.gamma
        E = len(hye_weights)
        N = u.shape[0]
        K = self.K

        # Numerator.
        if (X is None) or (gamma == 0):
            b_attributes = 0
            c = 0
        else:
            denominator = u @ beta
            assert denominator.shape == X.shape
            b_attributes = gamma * u * ((X / denominator) @ beta.T)

            u_reverse = 1 - u
            denominator = u_reverse @ beta
            assert denominator.shape == X.shape
            c = gamma * u_reverse * (((1 - X) / denominator) @ beta.T)

        if gamma == 1:
            return b_attributes / (b_attributes + c)
        else:
            poisson_params, edge_sum = self.poisson_params(
                binary_incidence, return_edge_sum=True
            )

            multiplier = hye_weights / poisson_params
            assert multiplier.shape == (E,)

            if sparse.issparse(binary_incidence):
                weighting = binary_incidence.multiply(multiplier[None, :])
                assert sparse.issparse(weighting)
            else:
                weighting = binary_incidence * multiplier[None, :]
            assert weighting.shape == (N, E)

            first_addend = weighting @ edge_sum
            assert first_addend.shape == (N, K)

            if sparse.issparse(weighting):
                weighting_sum = np.asarray(weighting.sum(axis=1)).reshape(-1, 1)
            else:
                weighting_sum = weighting.sum(axis=1, keepdims=True)
            second_addend = weighting_sum * u
            assert second_addend.shape == (N, K)

            b_hypergraph = (1 - gamma) * u * np.matmul(first_addend - second_addend, w)

            # Denominator
            u_sum = u.sum(axis=0)
            assert u_sum.shape == (K,)
            a = self.C() * (np.matmul(w, u_sum)[None, :] - np.matmul(u, w))
            assert a.shape == (N, K)

            if (X is None) or (gamma == 0):
                return b_hypergraph / a
            else:
                b = b_attributes + b_hypergraph
                b_equation = a + b + c
                D = b_equation**2 - 4 * a * b
                D = np.clip(D, a_min=0, a_max=None)
                x_1 = (b_equation - np.sqrt(D)) / (2 * a)
                if np.any(a == 0):
                    x_1[np.where(a == 0)] = (b / b_equation)[np.where(a == 0)]
                return x_1

    def _beta_update(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """EM updates for beta matrix."""
        u, beta = self.u, self.beta

        part1 = u @ beta
        assert part1.shape == X.shape
        numerator1 = beta * (u.T @ (X / part1))

        u_reverse = 1 - u
        part1 = u_reverse @ beta
        assert part1.shape == X.shape
        numerator2 = beta * (u_reverse.T @ ((1 - X) / part1))

        numerator = numerator1 + numerator2

        denominator = numerator.sum(axis=0, keepdims=True)
        return numerator / denominator

    def _check_beta_consistency(self) -> None:
        if self.beta is not None:
            if not self.beta.shape[0] == self.K:
                raise ValueError(
                    "The number of communities of beta and K are different"
                )

    def _check_attributes_consistency(self, X: Optional[np.ndarray] = None) -> None:
        if self.beta is not None and X is not None:
            if not self.beta.shape[1] == X.shape[1]:
                raise ValueError("The number of attributes of beta and X are different")

        if self.u is not None and X is not None:
            if X.shape[0] != self.u.shape[0]:
                raise ValueError("The number of nodes in X and u are different")
        if (X is None) and (self.gamma != 0):
            raise ValueError(
                "The attributes are not provided, but the gamma value is not 0."
            )

    def _init_beta(self, X: np.ndarray) -> None:
        K = self.K
        Z = X.shape[1]
        rng = self._rng
        beta = rng.random((K, Z))

        self.beta = beta / beta.sum(axis=0, keepdims=True)
