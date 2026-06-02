import numpy as np
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from itertools import combinations
from functools import lru_cache
import time
import hashlib


try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not found, parallel execution disabled")


try:
    from Code_DC_FNCD.federated_cf_ci_test_gpu import (
        FederatedCFCI_GPU,
        get_backend,
        LocalCFStats,
        NumpyBackend
    )
    FEDCFCIT_AVAILABLE = True
except ImportError:
    FEDCFCIT_AVAILABLE = False
    print("Warning: federated_cf_ci_test_gpu.py not found, using built-in implementation")


@dataclass
class LocalCFStatistics:
    client_id: int
    n_samples: int


    phi_X_real: np.ndarray
    phi_X_imag: np.ndarray
    phi_Y_real: np.ndarray
    phi_Y_imag: np.ndarray


    phi_XY_real: np.ndarray
    phi_XY_imag: np.ndarray


    second_moments: Dict = field(default_factory=dict)


@dataclass
class PermutationStatistics:
    client_id: int
    n_samples: int

    perm_phi_XY_real: np.ndarray
    perm_phi_XY_imag: np.ndarray


def _make_cache_key(var_X: int, var_Y: int, var_Z: List[int]) -> str:
    z_str = ','.join(map(str, sorted(var_Z))) if var_Z else ''
    return f"{var_X}:{var_Y}:{z_str}"


class PrivacyPreservingClient:

    def __init__(
        self,
        client_id: int,
        local_data: np.ndarray,
        n_frequencies: int = 100,
        frequency_scale: float = 1.0,
        n_neighbors: int = 15,
        backend: str = 'auto',
        random_state: int = 42,
        cache_size: int = 128
    ):
        self.client_id = client_id
        self._data = local_data
        self.n_samples = local_data.shape[0]
        self.n_vars = local_data.shape[1]
        self.n_frequencies = n_frequencies
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.cache_size = cache_size


        if FEDCFCIT_AVAILABLE:
            self.backend = get_backend(backend)
        else:
            self.backend = None


        np.random.seed(random_state)
        self.freq_s = np.random.randn(n_frequencies) * frequency_scale
        self.freq_t = np.random.randn(n_frequencies) * frequency_scale


        if self.backend is not None:
            self.freq_s_gpu = self.backend.array(self.freq_s)
            self.freq_t_gpu = self.backend.array(self.freq_t)


        self._residual_cache: Dict[str, np.ndarray] = {}
        self._cf_cache: Dict[str, Tuple] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_residual_cache_key(self, var_idx: int, var_Z: List[int]) -> str:
        z_str = ','.join(map(str, sorted(var_Z))) if var_Z else ''
        return f"res:{var_idx}:{z_str}"

    def _get_cached_residual(self, var_idx: int, var_Z: List[int]) -> Optional[np.ndarray]:
        key = self._get_residual_cache_key(var_idx, var_Z)
        if key in self._residual_cache:
            self._cache_hits += 1
            return self._residual_cache[key]
        self._cache_misses += 1
        return None

    def _set_cached_residual(self, var_idx: int, var_Z: List[int], residual: np.ndarray):

        if len(self._residual_cache) >= self.cache_size:
            keys_to_remove = list(self._residual_cache.keys())[:self.cache_size // 2]
            for k in keys_to_remove:
                del self._residual_cache[k]

        key = self._get_residual_cache_key(var_idx, var_Z)
        self._residual_cache[key] = residual

    def clear_cache(self):
        self._residual_cache.clear()
        self._cf_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'residual_cache_size': len(self._residual_cache),
            'cf_cache_size': len(self._cf_cache)
        }

    def compute_ci_statistics(
        self,
        var_X: int,
        var_Y: int,
        var_Z: List[int] = None
    ) -> LocalCFStatistics:
        if var_Z is None:
            var_Z = []


        X = self._data[:, var_X:var_X+1].astype(np.float32)
        Y = self._data[:, var_Y:var_Y+1].astype(np.float32)


        if var_Z:

            cached_X = self._get_cached_residual(var_X, var_Z)
            cached_Y = self._get_cached_residual(var_Y, var_Z)

            if cached_X is not None:
                X = cached_X
            else:
                Z = self._data[:, var_Z].astype(np.float32)
                X = self._compute_residuals(X, Z)
                self._set_cached_residual(var_X, var_Z, X)

            if cached_Y is not None:
                Y = cached_Y
            else:
                Z = self._data[:, var_Z].astype(np.float32)
                Y = self._compute_residuals(Y, Z)
                self._set_cached_residual(var_Y, var_Z, Y)


        if self.backend is not None and hasattr(self.backend, 'array'):
            phi_X_real, phi_X_imag = self._compute_cf_gpu(X, self.freq_s_gpu)
            phi_Y_real, phi_Y_imag = self._compute_cf_gpu(Y, self.freq_t_gpu)
            phi_XY_real, phi_XY_imag = self._compute_joint_cf_gpu(
                X, Y, self.freq_s_gpu, self.freq_t_gpu
            )
        else:
            phi_X_real, phi_X_imag = self._compute_cf_numpy(X, self.freq_s)
            phi_Y_real, phi_Y_imag = self._compute_cf_numpy(Y, self.freq_t)
            phi_XY_real, phi_XY_imag = self._compute_joint_cf_numpy(
                X, Y, self.freq_s, self.freq_t
            )


        second_moments = self._compute_second_moments(X, Y)

        return LocalCFStatistics(
            client_id=self.client_id,
            n_samples=self.n_samples,
            phi_X_real=phi_X_real,
            phi_X_imag=phi_X_imag,
            phi_Y_real=phi_Y_real,
            phi_Y_imag=phi_Y_imag,
            phi_XY_real=phi_XY_real,
            phi_XY_imag=phi_XY_imag,
            second_moments=second_moments
        )

    def compute_permutation_statistics(
            self,
            var_X: int,
            var_Y: int,
            var_Z: List[int] = None,
            n_permutations: int = 100,
            perm_seed: int = None
    ) -> PermutationStatistics:
        if var_Z is None:
            var_Z = []

        X = self._data[:, var_X:var_X+1].astype(np.float32)
        Y = self._data[:, var_Y:var_Y+1].astype(np.float32)


        if var_Z:
            cached_X = self._get_cached_residual(var_X, var_Z)
            cached_Y = self._get_cached_residual(var_Y, var_Z)

            if cached_X is not None:
                X = cached_X
            else:
                Z = self._data[:, var_Z].astype(np.float32)
                X = self._compute_residuals(X, Z)
                self._set_cached_residual(var_X, var_Z, X)

            if cached_Y is not None:
                Y = cached_Y
            else:
                Z = self._data[:, var_Z].astype(np.float32)
                Y = self._compute_residuals(Y, Z)
                self._set_cached_residual(var_Y, var_Z, Y)

        n = len(X)


        rng = np.random.RandomState(perm_seed if perm_seed is not None else self.random_state)


        perm_indices = np.array([rng.permutation(n) for _ in range(n_permutations)])


        if self.backend is not None and hasattr(self.backend, 'array'):
            perm_phi_XY_real, perm_phi_XY_imag = self._compute_joint_cf_vectorized_gpu(
                X, Y, perm_indices, self.freq_s_gpu, self.freq_t_gpu
            )
        else:
            perm_phi_XY_real, perm_phi_XY_imag = self._compute_joint_cf_vectorized_numpy(
                X, Y, perm_indices, self.freq_s, self.freq_t
            )

        return PermutationStatistics(
            client_id=self.client_id,
            n_samples=self.n_samples,
            perm_phi_XY_real=perm_phi_XY_real,
            perm_phi_XY_imag=perm_phi_XY_imag
        )

    def _compute_joint_cf_vectorized_numpy(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        perm_indices: np.ndarray,
        freq_s: np.ndarray,
        freq_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_permutations = perm_indices.shape[0]
        n_frequencies = len(freq_s)

        X_flat = X.flatten()
        Y_flat = Y.flatten()


        Y_perm_all = Y_flat[perm_indices]


        sX = np.outer(X_flat, freq_s)


        tY_perm_all = Y_perm_all[:, :, np.newaxis] * freq_t[np.newaxis, np.newaxis, :]


        arg = sX[np.newaxis, :, :] + tY_perm_all


        perm_phi_XY_real = np.mean(np.cos(arg), axis=1)
        perm_phi_XY_imag = np.mean(np.sin(arg), axis=1)

        return perm_phi_XY_real, perm_phi_XY_imag

    def _compute_joint_cf_vectorized_gpu(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        perm_indices: np.ndarray,
        freq_s_gpu,
        freq_t_gpu
    ) -> Tuple[np.ndarray, np.ndarray]:
        B = self.backend
        n_permutations = perm_indices.shape[0]
        n = len(X)

        X_flat = X.flatten()
        Y_flat = Y.flatten()


        X_gpu = B.array(X_flat)
        Y_gpu = B.array(Y_flat)
        freq_s = B.to_numpy(freq_s_gpu)
        freq_t = B.to_numpy(freq_t_gpu)


        batch_size = min(50, n_permutations)
        n_batches = (n_permutations + batch_size - 1) // batch_size

        all_real = []
        all_imag = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_permutations)
            batch_perm = perm_indices[start:end]


            Y_perm_batch = Y_flat[batch_perm]

            sX = np.outer(X_flat, freq_s)
            tY_batch = Y_perm_batch[:, :, np.newaxis] * freq_t[np.newaxis, np.newaxis, :]

            arg = sX[np.newaxis, :, :] + tY_batch

            batch_real = np.mean(np.cos(arg), axis=1)
            batch_imag = np.mean(np.sin(arg), axis=1)

            all_real.append(batch_real)
            all_imag.append(batch_imag)

        perm_phi_XY_real = np.vstack(all_real)
        perm_phi_XY_imag = np.vstack(all_imag)

        return perm_phi_XY_real, perm_phi_XY_imag

    def compute_anm_statistics(
        self,
        cause_var: int,
        effect_var: int
    ) -> LocalCFStatistics:
        cause = self._data[:, cause_var:cause_var+1].astype(np.float32)
        effect = self._data[:, effect_var:effect_var+1].astype(np.float32)


        residual = self._compute_regression_residual(cause, effect)


        if self.backend is not None:
            phi_res_real, phi_res_imag = self._compute_cf_gpu(residual, self.freq_s_gpu)
            phi_cause_real, phi_cause_imag = self._compute_cf_gpu(cause, self.freq_t_gpu)
            phi_joint_real, phi_joint_imag = self._compute_joint_cf_gpu(
                residual, cause, self.freq_s_gpu, self.freq_t_gpu
            )
        else:
            phi_res_real, phi_res_imag = self._compute_cf_numpy(residual, self.freq_s)
            phi_cause_real, phi_cause_imag = self._compute_cf_numpy(cause, self.freq_t)
            phi_joint_real, phi_joint_imag = self._compute_joint_cf_numpy(
                residual, cause, self.freq_s, self.freq_t
            )

        return LocalCFStatistics(
            client_id=self.client_id,
            n_samples=self.n_samples,
            phi_X_real=phi_res_real,
            phi_X_imag=phi_res_imag,
            phi_Y_real=phi_cause_real,
            phi_Y_imag=phi_cause_imag,
            phi_XY_real=phi_joint_real,
            phi_XY_imag=phi_joint_imag
        )


    def _compute_cf_gpu(self, X: np.ndarray, freqs) -> Tuple[np.ndarray, np.ndarray]:
        B = self.backend
        X_gpu = B.array(X.flatten())

        tX = X_gpu.reshape(-1, 1) * freqs.reshape(1, -1)
        real_part = B.to_numpy(B.mean(B.cos(tX), axis=0))
        imag_part = B.to_numpy(B.mean(B.sin(tX), axis=0))

        return real_part, imag_part

    def _compute_joint_cf_gpu(
        self, X: np.ndarray, Y: np.ndarray, freq_s, freq_t
    ) -> Tuple[np.ndarray, np.ndarray]:
        B = self.backend
        X_gpu = B.array(X.flatten())
        Y_gpu = B.array(Y.flatten())

        sX = X_gpu.reshape(-1, 1) * freq_s.reshape(1, -1)
        tY = Y_gpu.reshape(-1, 1) * freq_t.reshape(1, -1)
        arg = sX + tY

        real_part = B.to_numpy(B.mean(B.cos(arg), axis=0))
        imag_part = B.to_numpy(B.mean(B.sin(arg), axis=0))

        return real_part, imag_part

    def _compute_cf_numpy(self, X: np.ndarray, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_flat = X.flatten()
        tX = np.outer(X_flat, freqs)
        real_part = np.mean(np.cos(tX), axis=0)
        imag_part = np.mean(np.sin(tX), axis=0)
        return real_part, imag_part

    def _compute_joint_cf_numpy(
        self, X: np.ndarray, Y: np.ndarray,
        freq_s: np.ndarray, freq_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        sX = np.outer(X_flat, freq_s)
        tY = np.outer(Y_flat, freq_t)
        arg = sX + tY
        real_part = np.mean(np.cos(arg), axis=0)
        imag_part = np.mean(np.sin(arg), axis=0)
        return real_part, imag_part

    def _compute_second_moments(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        sX = np.outer(X_flat, self.freq_s)
        tY = np.outer(Y_flat, self.freq_t)

        return {
            'cos2_sX': np.mean(np.cos(sX) ** 2, axis=0),
            'sin2_sX': np.mean(np.sin(sX) ** 2, axis=0),
            'cos2_tY': np.mean(np.cos(tY) ** 2, axis=0),
            'sin2_tY': np.mean(np.sin(tY) ** 2, axis=0),
        }

    def _compute_residuals(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        from sklearn.neighbors import KNeighborsRegressor

        n = len(X)
        k = min(self.n_neighbors, n - 1)

        residuals = np.zeros_like(X)
        for j in range(X.shape[1]):

            knn = KNeighborsRegressor(n_neighbors=k, algorithm='ball_tree', n_jobs=1)
            knn.fit(Z, X[:, j])
            residuals[:, j] = X[:, j] - knn.predict(Z)

        return residuals

    def _compute_regression_residual(
        self, cause: np.ndarray, effect: np.ndarray
    ) -> np.ndarray:
        from sklearn.neighbors import KNeighborsRegressor

        k = min(self.n_neighbors, len(cause) - 1)
        knn = KNeighborsRegressor(n_neighbors=k, algorithm='ball_tree', n_jobs=1)
        knn.fit(cause, effect.ravel())

        return effect - knn.predict(cause).reshape(-1, 1)


class PrivacyPreservingServer:

    def __init__(
        self,
        clients: List[PrivacyPreservingClient],
        alpha: float = 0.01,
        n_permutations: int = 500,
        use_permutation_test: bool = True,
        random_state: int = 66,
        n_jobs: int = -1
    ):
        self.clients = clients
        self.n_clients = len(clients)
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.use_permutation_test = use_permutation_test
        self.random_state = random_state
        self.n_jobs = n_jobs if JOBLIB_AVAILABLE else 1

        self.n_vars = clients[0].n_vars
        self.n_frequencies = clients[0].n_frequencies
        self.freq_s = clients[0].freq_s
        self.freq_t = clients[0].freq_t


        self._ci_cache: Dict[str, Tuple[float, float, bool]] = {}

    def _get_ci_cache_key(self, var_X: int, var_Y: int, var_Z: List[int]) -> str:
        z_str = ','.join(map(str, sorted(var_Z))) if var_Z else ''

        if var_X > var_Y:
            var_X, var_Y = var_Y, var_X
        return f"ci:{var_X}:{var_Y}:{z_str}"

    def clear_cache(self):
        self._ci_cache.clear()

    def federated_ci_test(
        self,
        var_X: int,
        var_Y: int,
        var_Z: List[int] = None
    ) -> Tuple[float, float, bool]:
        if var_Z is None:
            var_Z = []


        cache_key = self._get_ci_cache_key(var_X, var_Y, var_Z)
        if cache_key in self._ci_cache:
            return self._ci_cache[cache_key]


        if JOBLIB_AVAILABLE and self.n_clients > 1 and self.n_jobs != 1:
            local_stats = Parallel(n_jobs=self.n_jobs)(
                delayed(client.compute_ci_statistics)(var_X, var_Y, var_Z)
                for client in self.clients
            )
        else:
            local_stats = [
                client.compute_ci_statistics(var_X, var_Y, var_Z)
                for client in self.clients
            ]


        n_total = sum(s.n_samples for s in local_stats)

        phi_X = np.zeros(self.n_frequencies, dtype=complex)
        phi_Y = np.zeros(self.n_frequencies, dtype=complex)
        phi_XY = np.zeros(self.n_frequencies, dtype=complex)

        for s in local_stats:
            w = s.n_samples / n_total
            phi_X += w * (s.phi_X_real + 1j * s.phi_X_imag)
            phi_Y += w * (s.phi_Y_real + 1j * s.phi_Y_imag)
            phi_XY += w * (s.phi_XY_real + 1j * s.phi_XY_imag)


        delta = phi_XY - phi_X * phi_Y
        cf_distance_sq = np.mean(np.abs(delta) ** 2)
        statistic = n_total * cf_distance_sq


        if self.use_permutation_test:
            pvalue = self._federated_permutation_test(
                var_X, var_Y, var_Z, statistic, phi_X, phi_Y
            )
        else:
            pvalue = self._asymptotic_pvalue(statistic, n_total)

        is_independent = pvalue > self.alpha


        result = (pvalue, statistic, is_independent)
        self._ci_cache[cache_key] = result

        return result

    def get_anm_statistic(self, cause_var: int, effect_var: int) -> float:

        if JOBLIB_AVAILABLE and self.n_clients > 1 and self.n_jobs != 1:
            local_stats = Parallel(n_jobs=self.n_jobs)(
                delayed(client.compute_anm_statistics)(cause_var, effect_var)
                for client in self.clients
            )
        else:
            local_stats = [
                client.compute_anm_statistics(cause_var, effect_var)
                for client in self.clients
            ]

        n_total = sum(s.n_samples for s in local_stats)

        phi_res = np.zeros(self.n_frequencies, dtype=complex)
        phi_cause = np.zeros(self.n_frequencies, dtype=complex)
        phi_joint = np.zeros(self.n_frequencies, dtype=complex)

        for s in local_stats:
            w = s.n_samples / n_total
            phi_res += w * (s.phi_X_real + 1j * s.phi_X_imag)
            phi_cause += w * (s.phi_Y_real + 1j * s.phi_Y_imag)
            phi_joint += w * (s.phi_XY_real + 1j * s.phi_XY_imag)

        delta = phi_joint - phi_res * phi_cause
        cf_distance_sq = np.mean(np.abs(delta) ** 2)
        statistic = n_total * cf_distance_sq

        return statistic

    def _federated_permutation_test(
            self,
            var_X: int,
            var_Y: int,
            var_Z: List[int],
            observed_statistic: float,
            phi_X_cached: np.ndarray,
            phi_Y_cached: np.ndarray
    ) -> float:
        n_total = sum(c.n_samples for c in self.clients)


        if JOBLIB_AVAILABLE and self.n_clients > 1 and self.n_jobs != 1:
            perm_stats_list = Parallel(n_jobs=self.n_jobs)(
                delayed(client.compute_permutation_statistics)(
                    var_X, var_Y, var_Z, self.n_permutations,
                    perm_seed=self.random_state + idx
                )
                for idx, client in enumerate(self.clients)
            )
        else:
            perm_stats_list = []
            for idx, client in enumerate(self.clients):
                perm_stats = client.compute_permutation_statistics(
                    var_X, var_Y, var_Z, self.n_permutations,
                    perm_seed=self.random_state + idx
                )
                perm_stats_list.append(perm_stats)


        phi_XY_perm_all = np.zeros((self.n_permutations, self.n_frequencies), dtype=complex)

        for ps in perm_stats_list:
            w = ps.n_samples / n_total
            phi_XY_perm_all += w * (ps.perm_phi_XY_real + 1j * ps.perm_phi_XY_imag)


        delta_perm_all = phi_XY_perm_all - phi_X_cached * phi_Y_cached
        perm_statistics = n_total * np.mean(np.abs(delta_perm_all) ** 2, axis=1)


        pvalue = (np.sum(perm_statistics >= observed_statistic) + 1) / (self.n_permutations + 1)

        return float(pvalue)

    def _asymptotic_pvalue(self, statistic: float, n: int) -> float:
        from scipy.stats import gamma

        k = self.n_frequencies / 2
        theta = 2 / n
        pvalue = 1 - gamma.cdf(statistic, k, scale=theta)

        return float(pvalue)

    def federated_anm_test(self, cause_var: int, effect_var: int) -> Tuple[float, float]:

        if JOBLIB_AVAILABLE and self.n_clients > 1 and self.n_jobs != 1:
            local_stats = Parallel(n_jobs=self.n_jobs)(
                delayed(client.compute_anm_statistics)(cause_var, effect_var)
                for client in self.clients
            )
        else:
            local_stats = [
                client.compute_anm_statistics(cause_var, effect_var)
                for client in self.clients
            ]


        n_total = sum(s.n_samples for s in local_stats)

        phi_res = np.zeros(self.n_frequencies, dtype=complex)
        phi_cause = np.zeros(self.n_frequencies, dtype=complex)
        phi_joint = np.zeros(self.n_frequencies, dtype=complex)

        for s in local_stats:
            w = s.n_samples / n_total
            phi_res += w * (s.phi_X_real + 1j * s.phi_X_imag)
            phi_cause += w * (s.phi_Y_real + 1j * s.phi_Y_imag)
            phi_joint += w * (s.phi_XY_real + 1j * s.phi_XY_imag)


        delta = phi_joint - phi_res * phi_cause
        cf_distance_sq = np.mean(np.abs(delta) ** 2)
        statistic = n_total * cf_distance_sq

        pvalue = self._asymptotic_pvalue(statistic, n_total)

        return pvalue, statistic


class IntegratedFederatedCausalDiscovery:

    def __init__(
        self,
        clients: List[PrivacyPreservingClient],
        alpha: float = 0.01,
        max_conditioning_size: int = 3,
        skeleton_strategy: str = 'confirm',
        use_anm: bool = True,
        use_permutation_test: bool = True,
        n_permutations: int = 200,
        random_state: int = 66,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        self.server = PrivacyPreservingServer(
            clients=clients,
            alpha=alpha,
            n_permutations=n_permutations,
            use_permutation_test=use_permutation_test,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.alpha = alpha
        self.max_k = max_conditioning_size
        self.strategy = skeleton_strategy
        self.use_anm = use_anm
        self.verbose = verbose
        self.n_vars = clients[0].n_vars

        self.n_ci_tests = 0

    def discover(self) -> Dict:
        start_time = time.time()

        if self.verbose:
            print("\n" + "="*70)
            print("INTEGRATED FEDERATED CAUSAL DISCOVERY (OPTIMIZED)")
            print("="*70)
            print(f"Clients: {self.server.n_clients}")
            print(f"Variables: {self.n_vars}")
            print(f"Backend: {self.server.clients[0].backend.name if self.server.clients[0].backend else 'numpy'}")
            print(f"Parallel: {'enabled (joblib)' if JOBLIB_AVAILABLE and self.server.n_jobs != 1 else 'disabled'}")
            print(f"Privacy: Data remains on clients")
            print("="*70)


        pc_results, sepset_all = self._phase1_pc_learning()


        skeleton, global_sepset = self._phase2_skeleton_merge(pc_results, sepset_all)


        cpdag = self._phase3_orientation(skeleton, global_sepset)

        total_time = time.time() - start_time


        if self.verbose:
            print(f"\n{'='*70}")
            print(f"COMPLETE")
            print(f"Total time: {total_time:.3f}s")
            print(f"Total CI tests: {self.n_ci_tests}")


            total_hits = 0
            total_misses = 0
            for client in self.server.clients:
                stats = client.get_cache_stats()
                total_hits += stats['hits']
                total_misses += stats['misses']

            if total_hits + total_misses > 0:
                print(f"Cache hit rate: {total_hits / (total_hits + total_misses):.2%}")

            print(f"{'='*70}")

        return {
            'cpdag': cpdag,
            'skeleton': skeleton,
            'sepset': global_sepset,
            'n_ci_tests': self.n_ci_tests,
            'total_time': total_time
        }

    def _phase1_pc_learning(self) -> Tuple[Dict, Dict]:
        if self.verbose:
            print("\n[Phase 1] Federated PC Learning")

        pc_results = {}
        sepset_all = {}

        for target in range(self.n_vars):
            pc, sepset = self._learn_pc(target)
            pc_results[target] = pc
            sepset_all[target] = sepset

            if self.verbose:
                print(f"  PC({target}) = {pc}")

        return pc_results, sepset_all

    def _learn_pc(self, target: int) -> Tuple[List[int], Dict]:
        cpc = []
        dep = {}
        sepset = {i: [] for i in range(self.n_vars)}


        for i in range(self.n_vars):
            if i == target:
                continue

            self.n_ci_tests += 1
            pvalue, stat, is_indep = self.server.federated_ci_test(i, target, [])

            if not is_indep:
                cpc.append(i)
                dep[i] = stat

        if not cpc:
            return [], sepset


        var_order = sorted(cpc, key=lambda x: dep.get(x, 0), reverse=True)

        pc = []
        for Y in var_order:
            pc.append(Y)
            pc_tmp = pc.copy()

            last_break_flag = False

            for X in reversed(pc):
                CanPC = [v for v in pc_tmp if v != X]
                found = False

                for size in range(min(len(CanPC) + 1, self.max_k + 1)):
                    for Z in combinations(CanPC, size):
                        Z = list(Z)
                        if X != Y and Y not in Z:
                            continue

                        if not Z:
                            continue

                        self.n_ci_tests += 1
                        pvalue, stat, is_indep = self.server.federated_ci_test(X, target, Z)

                        if is_indep:
                            pc_tmp = CanPC
                            sepset[X] = Z
                            found = True

                            if X == Y:
                                last_break_flag = True

                            break

                    if found:
                        break
                    if last_break_flag:
                        break

                if last_break_flag:
                    break

            pc = pc_tmp

        return pc, sepset

    def _phase2_skeleton_merge(
            self, pc_results: Dict, sepset_all: Dict
    ) -> Tuple[np.ndarray, Dict]:
        if self.verbose:
            print(f"\n[Phase 2] Skeleton Merge (Strategy: {self.strategy})")

        p = self.n_vars
        skeleton = np.zeros((p, p), dtype=int)
        global_sepset = {}


        symmetric = set()
        asymmetric = set()

        for i in range(p):
            for j in pc_results[i]:
                edge = (min(i, j), max(i, j))
                if i in pc_results[j]:
                    symmetric.add(edge)
                else:
                    asymmetric.add(edge)


        for (i, j) in symmetric:
            skeleton[i, j] = skeleton[j, i] = 1


        if self.strategy == 'confirm':
            for (i, j) in asymmetric:
                stat_ij = self.server.get_anm_statistic(i, j)
                stat_ji = self.server.get_anm_statistic(j, i)

                min_stat = min(stat_ij, stat_ji)
                max_stat = max(stat_ij, stat_ji)

                if max_stat > min_stat * 2:
                    skeleton[i, j] = skeleton[j, i] = 1
                else:
                    global_sepset[(i, j)] = []
                    global_sepset[(j, i)] = []

        elif self.strategy == 'or':
            for (i, j) in asymmetric:
                skeleton[i, j] = skeleton[j, i] = 1


        for target in range(p):
            for var, sep in sepset_all[target].items():
                if sep:
                    if (var, target) not in global_sepset:
                        global_sepset[(var, target)] = sep

        if self.verbose:
            print(f"  Skeleton edges: {int(np.sum(skeleton) / 2)}")

        return skeleton, global_sepset

    def _phase3_orientation(self, skeleton: np.ndarray, sepset: Dict) -> Dict:
        if self.verbose:
            print("\n[Phase 3] Orientation")

        p = skeleton.shape[0]
        direction = np.zeros((p, p), dtype=int)


        v_structures = []


        if self.use_anm:
            undirected = []
            for i in range(p):
                for j in range(i + 1, p):
                    if skeleton[i, j] == 1 and direction[i, j] == 0 and direction[j, i] == 0:
                        undirected.append((i, j))

            if undirected and self.verbose:
                print(f"  ANM orienting {len(undirected)} edges...")

            for (i, j) in undirected:
                _, score_ij = self.server.federated_anm_test(i, j)
                _, score_ji = self.server.federated_anm_test(j, i)


                if score_ij <= score_ji:
                    direction[i, j] = 1
                else:
                    direction[j, i] = 1


        directed_edges = []
        undirected_edges = []

        for i in range(p):
            for j in range(i + 1, p):
                if skeleton[i, j] == 1:
                    if direction[i, j] == 1:
                        directed_edges.append((i, j))
                    elif direction[j, i] == 1:
                        directed_edges.append((j, i))
                    else:
                        undirected_edges.append((i, j))

        if self.verbose:
            print(f"  Directed: {len(directed_edges)}, Undirected: {len(undirected_edges)}")

        return {
            'directed_edges': directed_edges,
            'undirected_edges': undirected_edges,
            'v_structures': v_structures,
            'direction_matrix': direction
        }

    def _apply_meek_rules(self, skeleton: np.ndarray, direction: np.ndarray) -> np.ndarray:
        p = skeleton.shape[0]
        changed = True

        while changed:
            changed = False


            for X in range(p):
                for Y in range(p):
                    if direction[X, Y] == 1:
                        for Z in range(p):
                            if (Z != X and
                                    skeleton[Y, Z] == 1 and
                                    direction[Y, Z] == 0 and direction[Z, Y] == 0 and
                                    skeleton[X, Z] == 0):
                                direction[Y, Z] = 1
                                changed = True


            for X in range(p):
                for Z in range(p):
                    if (skeleton[X, Z] == 1 and
                            direction[X, Z] == 0 and direction[Z, X] == 0):
                        for Y in range(p):
                            if direction[X, Y] == 1 and direction[Y, Z] == 1:
                                direction[X, Z] = 1
                                changed = True
                                break


            for X in range(p):
                for W in range(p):
                    if (skeleton[X, W] == 1 and
                            direction[X, W] == 0 and direction[W, X] == 0):
                        neighbors = [n for n in range(p)
                                     if skeleton[X, n] == 1 and
                                     direction[X, n] == 0 and direction[n, X] == 0]
                        for i, Y in enumerate(neighbors):
                            for Z in neighbors[i + 1:]:
                                if (direction[Y, W] == 1 and direction[Z, W] == 1 and
                                        skeleton[Y, Z] == 0):
                                    direction[X, W] = 1
                                    changed = True
                                    break
                            if direction[X, W] == 1:
                                break


            for X in range(p):
                for W in range(p):
                    if (skeleton[X, W] == 1 and
                            direction[X, W] == 0 and direction[W, X] == 0):
                        for Y in range(p):
                            if (skeleton[X, Y] == 1 and
                                    direction[X, Y] == 0 and direction[Y, X] == 0):
                                for Z in range(p):
                                    if (skeleton[X, Z] == 1 and
                                            direction[Y, Z] == 1 and
                                            direction[Z, W] == 1):
                                        direction[X, W] = 1
                                        changed = True
                                        break
                            if direction[X, W] == 1:
                                break

        return direction


def create_federated_clients(
    data_list: List[np.ndarray],
    n_frequencies: int = 100,
    backend: str = 'auto',
    random_state: int = 42,
    cache_size: int = 128
) -> List[PrivacyPreservingClient]:
    clients = []
    for k, data in enumerate(data_list):
        client = PrivacyPreservingClient(
            client_id=k,
            local_data=data,
            n_frequencies=n_frequencies,
            backend=backend,
            random_state=random_state,
            cache_size=cache_size
        )
        clients.append(client)

    return clients


def dc_fncd(
    data_list: List[np.ndarray],
    alpha: float = 0.01,
    max_k: int = 3,
    n_permutations: int = 200,
    backend: str = 'auto',
    random_state: int = 123,
    n_jobs: int = -1,
    verbose: bool = True
) -> Dict:

    clients = create_federated_clients(
        data_list=data_list,
        n_frequencies=n_permutations,
        backend=backend,
        random_state=random_state
    )


    discoverer = IntegratedFederatedCausalDiscovery(
        clients=clients,
        alpha=alpha,
        max_conditioning_size=max_k,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )


    return discoverer.discover()
