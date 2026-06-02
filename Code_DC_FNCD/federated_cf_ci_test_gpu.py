import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
import warnings
import time


class ComputeBackend:

    def __init__(self):
        self.name = "base"
        self.device = "cpu"

    def array(self, x):
        raise NotImplementedError

    def to_numpy(self, x):
        raise NotImplementedError

    def cos(self, x):
        raise NotImplementedError

    def sin(self, x):
        raise NotImplementedError

    def exp(self, x):
        raise NotImplementedError

    def abs(self, x):
        raise NotImplementedError

    def mean(self, x, axis=None):
        raise NotImplementedError

    def sum(self, x, axis=None):
        raise NotImplementedError

    def randn(self, *shape):
        raise NotImplementedError

    def permutation(self, n):
        raise NotImplementedError

    def matmul(self, a, b):
        raise NotImplementedError

    def norm(self, x, axis=None):
        raise NotImplementedError

    def zeros(self, shape):
        raise NotImplementedError

    def hstack(self, arrays):
        raise NotImplementedError

    def vstack(self, arrays):
        raise NotImplementedError


class NumpyBackend(ComputeBackend):

    def __init__(self):
        super().__init__()
        self.name = "numpy"
        self.device = "cpu"
        self.xp = np

    def array(self, x):
        return np.asarray(x)

    def to_numpy(self, x):
        return np.asarray(x)

    def cos(self, x):
        return np.cos(x)

    def sin(self, x):
        return np.sin(x)

    def exp(self, x):
        return np.exp(x)

    def abs(self, x):
        return np.abs(x)

    def mean(self, x, axis=None):
        return np.mean(x, axis=axis)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def randn(self, *shape):
        return np.random.randn(*shape)

    def permutation(self, n):
        return np.random.permutation(n)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def norm(self, x, axis=None, keepdims=False):
        return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    def zeros(self, shape):
        return np.zeros(shape)

    def hstack(self, arrays):
        return np.hstack(arrays)

    def vstack(self, arrays):
        return np.vstack(arrays)


class CupyBackend(ComputeBackend):

    def __init__(self):
        super().__init__()
        try:
            import cupy as cp
            self.xp = cp
            self.name = "cupy"
            self.device = f"cuda:{cp.cuda.Device().id}"


            test_arr = cp.zeros(100)
            _ = test_arr * 2
            cp.cuda.Stream.null.synchronize()

        except ImportError:
            raise ImportError("CuPy is not installed. Please run: pip install cupy-cuda11x (choose the one matching your CUDA version)")
        except (RuntimeError, OSError) as e:
            raise RuntimeError(
                f"CuPy initialization failed: {e}\n"
                f"This is usually a CUDA path issue. Please try:\n"
                f"1. Set the CUDA_PATH environment variable to your CUDA installation directory\n"
                f"2. Or use the PyTorch backend: backend='torch'"
            )

    def array(self, x):
        return self.xp.asarray(x)

    def to_numpy(self, x):
        return self.xp.asnumpy(x)

    def cos(self, x):
        return self.xp.cos(x)

    def sin(self, x):
        return self.xp.sin(x)

    def exp(self, x):
        return self.xp.exp(x)

    def abs(self, x):
        return self.xp.abs(x)

    def mean(self, x, axis=None):
        return self.xp.mean(x, axis=axis)

    def sum(self, x, axis=None):
        return self.xp.sum(x, axis=axis)

    def randn(self, *shape):
        return self.xp.random.randn(*shape)

    def permutation(self, n):
        return self.xp.random.permutation(n)

    def matmul(self, a, b):
        return self.xp.matmul(a, b)

    def norm(self, x, axis=None, keepdims=False):
        return self.xp.linalg.norm(x, axis=axis, keepdims=keepdims)

    def zeros(self, shape):
        return self.xp.zeros(shape)

    def hstack(self, arrays):
        return self.xp.hstack(arrays)

    def vstack(self, arrays):
        return self.xp.vstack(arrays)

    def synchronize(self):
        self.xp.cuda.Stream.null.synchronize()


class TorchBackend(ComputeBackend):

    def __init__(self, device=None):
        super().__init__()
        try:
            import torch
            self.torch = torch
            self.name = "torch"

            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            self.torch_device = torch.device(self.device)


            if self.device == "cuda":
                _ = torch.zeros(100, device=self.torch_device)
                torch.cuda.synchronize()

        except ImportError:
            raise ImportError("PyTorch is not installed. Please run: pip install torch")

    def array(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.to(self.torch_device)
        return self.torch.tensor(x, dtype=self.torch.float32, device=self.torch_device)

    def to_numpy(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    def cos(self, x):
        return self.torch.cos(x)

    def sin(self, x):
        return self.torch.sin(x)

    def exp(self, x):
        return self.torch.exp(x)

    def abs(self, x):
        return self.torch.abs(x)

    def mean(self, x, axis=None):
        if axis is None:
            return self.torch.mean(x)
        return self.torch.mean(x, dim=axis)

    def sum(self, x, axis=None):
        if axis is None:
            return self.torch.sum(x)
        return self.torch.sum(x, dim=axis)

    def randn(self, *shape):
        return self.torch.randn(*shape, device=self.torch_device)

    def permutation(self, n):
        return self.torch.randperm(n, device=self.torch_device)

    def matmul(self, a, b):
        return self.torch.matmul(a, b)

    def norm(self, x, axis=None, keepdims=False):
        if axis is None:
            return self.torch.norm(x)
        return self.torch.norm(x, dim=axis, keepdim=keepdims)

    def zeros(self, shape):
        return self.torch.zeros(shape, device=self.torch_device)

    def hstack(self, arrays):
        return self.torch.hstack(arrays)

    def vstack(self, arrays):
        return self.torch.vstack(arrays)

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()


def get_backend(backend: str = 'auto') -> ComputeBackend:
    if backend == 'auto':

        try:
            import torch
            if torch.cuda.is_available():
                return TorchBackend(device='cuda')
        except ImportError:
            pass


        try:
            return CupyBackend()
        except (ImportError, RuntimeError, OSError) as e:

            pass


        return NumpyBackend()

    elif backend == 'cupy':
        return CupyBackend()

    elif backend == 'torch':
        return TorchBackend()

    elif backend == 'numpy':
        return NumpyBackend()

    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: auto, cupy, torch, numpy")


@dataclass
class LocalCFStats:
    n_samples: int
    phi_X_real: np.ndarray
    phi_X_imag: np.ndarray
    phi_Y_real: np.ndarray
    phi_Y_imag: np.ndarray
    phi_XY_real: np.ndarray
    phi_XY_imag: np.ndarray
    second_moments: Dict[str, np.ndarray] = field(default_factory=dict)


class FederatedCFCI_GPU:

    def __init__(
        self,
        n_frequencies: int = 100,
        frequency_scale: float = 1.0,
        n_neighbors: int = 15,
        alpha: float = 0.01,
        use_permutation: bool = True,
        n_permutations: int = 500,
        batch_permutations: int = 50,
        random_state: int = 66,
        backend: str = 'auto'
    ):
        self.n_frequencies = n_frequencies
        self.frequency_scale = frequency_scale
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.use_permutation = use_permutation
        self.n_permutations = n_permutations
        self.batch_permutations = batch_permutations
        self.random_state = random_state


        self.backend = get_backend(backend)


        np.random.seed(random_state)
        self.freq_s = np.random.randn(n_frequencies) * frequency_scale
        self.freq_t = np.random.randn(n_frequencies) * frequency_scale


        self.freq_s_gpu = self.backend.array(self.freq_s)
        self.freq_t_gpu = self.backend.array(self.freq_t)

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def _compute_empirical_cf_gpu(
        self,
        X,
        freqs
    ) -> Tuple:
        B = self.backend


        if len(X.shape) == 1 or X.shape[1] == 1:
            X_flat = X.reshape(-1) if hasattr(X, 'reshape') else X.flatten()

            tX = X_flat.reshape(-1, 1) * freqs.reshape(1, -1)
        else:

            n, d = X.shape
            M = len(freqs)


            np.random.seed(self.random_state + 1000)
            directions = np.random.randn(d, M)
            directions = directions / np.linalg.norm(directions, axis=0, keepdims=True)
            directions_gpu = B.array(directions)


            projected = B.matmul(X, directions_gpu)
            tX = projected * freqs.reshape(1, -1)


        real_part = B.mean(B.cos(tX), axis=0)
        imag_part = B.mean(B.sin(tX), axis=0)

        return real_part, imag_part

    def _compute_joint_cf_gpu(
        self,
        X,
        Y,
        freq_s,
        freq_t
    ) -> Tuple:
        B = self.backend

        X_flat = X.reshape(-1) if len(X.shape) > 1 and X.shape[1] == 1 else X
        Y_flat = Y.reshape(-1) if len(Y.shape) > 1 and Y.shape[1] == 1 else Y

        if len(X_flat.shape) == 1:
            sX = X_flat.reshape(-1, 1) * freq_s.reshape(1, -1)
        else:

            np.random.seed(self.random_state + 2000)
            d = X_flat.shape[1]
            M = len(freq_s)
            dir_X = np.random.randn(d, M)
            dir_X = dir_X / np.linalg.norm(dir_X, axis=0, keepdims=True)
            dir_X_gpu = B.array(dir_X)
            sX = B.matmul(X_flat, dir_X_gpu) * freq_s.reshape(1, -1)

        if len(Y_flat.shape) == 1:
            tY = Y_flat.reshape(-1, 1) * freq_t.reshape(1, -1)
        else:

            np.random.seed(self.random_state + 3000)
            d = Y_flat.shape[1]
            M = len(freq_t)
            dir_Y = np.random.randn(d, M)
            dir_Y = dir_Y / np.linalg.norm(dir_Y, axis=0, keepdims=True)
            dir_Y_gpu = B.array(dir_Y)
            tY = B.matmul(Y_flat, dir_Y_gpu) * freq_t.reshape(1, -1)


        arg = sX + tY
        real_part = B.mean(B.cos(arg), axis=0)
        imag_part = B.mean(B.sin(arg), axis=0)

        return real_part, imag_part

    def _compute_residuals_cpu(
        self,
        X: np.ndarray,
        Z: np.ndarray
    ) -> np.ndarray:
        from sklearn.neighbors import KNeighborsRegressor

        X = self._ensure_2d(X)
        Z = self._ensure_2d(Z)

        n = len(X)
        d_X = X.shape[1]
        k = min(self.n_neighbors, n - 1)

        residuals = np.zeros_like(X)

        for j in range(d_X):
            knn = KNeighborsRegressor(n_neighbors=k, weights='uniform')
            knn.fit(Z, X[:, j])
            X_pred = knn.predict(Z)
            residuals[:, j] = X[:, j] - X_pred

        return residuals

    def compute_local_statistics(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None
    ) -> LocalCFStats:
        B = self.backend

        X = self._ensure_2d(X)
        Y = self._ensure_2d(Y)
        n_k = len(X)


        if Z is not None:
            Z = self._ensure_2d(Z)
            X_use = self._compute_residuals_cpu(X, Z)
            Y_use = self._compute_residuals_cpu(Y, Z)
        else:
            X_use = X
            Y_use = Y


        X_gpu = B.array(X_use.astype(np.float32))
        Y_gpu = B.array(Y_use.astype(np.float32))


        phi_X_real, phi_X_imag = self._compute_empirical_cf_gpu(X_gpu, self.freq_s_gpu)
        phi_Y_real, phi_Y_imag = self._compute_empirical_cf_gpu(Y_gpu, self.freq_t_gpu)
        phi_XY_real, phi_XY_imag = self._compute_joint_cf_gpu(
            X_gpu, Y_gpu, self.freq_s_gpu, self.freq_t_gpu
        )


        X_flat = X_gpu.reshape(-1) if len(X_gpu.shape) > 1 else X_gpu
        Y_flat = Y_gpu.reshape(-1) if len(Y_gpu.shape) > 1 else Y_gpu

        sX = X_flat.reshape(-1, 1) * self.freq_s_gpu.reshape(1, -1)
        tY = Y_flat.reshape(-1, 1) * self.freq_t_gpu.reshape(1, -1)

        second_moments = {
            'cos2_sX': B.to_numpy(B.mean(B.cos(sX) ** 2, axis=0)),
            'sin2_sX': B.to_numpy(B.mean(B.sin(sX) ** 2, axis=0)),
            'cos2_tY': B.to_numpy(B.mean(B.cos(tY) ** 2, axis=0)),
            'sin2_tY': B.to_numpy(B.mean(B.sin(tY) ** 2, axis=0)),
        }


        if hasattr(B, 'synchronize'):
            B.synchronize()

        return LocalCFStats(
            n_samples=n_k,
            phi_X_real=B.to_numpy(phi_X_real),
            phi_X_imag=B.to_numpy(phi_X_imag),
            phi_Y_real=B.to_numpy(phi_Y_real),
            phi_Y_imag=B.to_numpy(phi_Y_imag),
            phi_XY_real=B.to_numpy(phi_XY_real),
            phi_XY_imag=B.to_numpy(phi_XY_imag),
            second_moments=second_moments
        )

    def aggregate_statistics(
        self,
        local_stats_list: List[LocalCFStats]
    ) -> Dict:
        n_total = sum(ls.n_samples for ls in local_stats_list)
        M = self.n_frequencies

        phi_X_real = np.zeros(M)
        phi_X_imag = np.zeros(M)
        phi_Y_real = np.zeros(M)
        phi_Y_imag = np.zeros(M)
        phi_XY_real = np.zeros(M)
        phi_XY_imag = np.zeros(M)

        for ls in local_stats_list:
            w_k = ls.n_samples / n_total
            phi_X_real += w_k * ls.phi_X_real
            phi_X_imag += w_k * ls.phi_X_imag
            phi_Y_real += w_k * ls.phi_Y_real
            phi_Y_imag += w_k * ls.phi_Y_imag
            phi_XY_real += w_k * ls.phi_XY_real
            phi_XY_imag += w_k * ls.phi_XY_imag

        phi_X = phi_X_real + 1j * phi_X_imag
        phi_Y = phi_Y_real + 1j * phi_Y_imag
        phi_XY = phi_XY_real + 1j * phi_XY_imag

        return {
            'n_total': n_total,
            'phi_X': phi_X,
            'phi_Y': phi_Y,
            'phi_XY': phi_XY,
        }

    def compute_test_statistic(self, global_stats: Dict) -> Tuple[float, float]:
        n = global_stats['n_total']
        phi_X = global_stats['phi_X']
        phi_Y = global_stats['phi_Y']
        phi_XY = global_stats['phi_XY']

        delta = phi_XY - phi_X * phi_Y
        delta_squared = np.abs(delta) ** 2
        cf_distance_squared = delta_squared.mean()
        statistic = n * cf_distance_squared

        return float(statistic), float(cf_distance_squared)

    def _batch_permutation_test_gpu(
        self,
        X_all: np.ndarray,
        Y_all: np.ndarray,
        observed_statistic: float
    ) -> float:
        B = self.backend
        n = len(X_all)
        M = self.n_frequencies


        X_gpu = B.array(X_all.flatten().astype(np.float32))
        Y_gpu = B.array(Y_all.flatten().astype(np.float32))

        perm_statistics = []
        n_batches = (self.n_permutations + self.batch_permutations - 1) // self.batch_permutations

        for batch_idx in range(n_batches):
            batch_size = min(self.batch_permutations,
                           self.n_permutations - batch_idx * self.batch_permutations)


            batch_stats = []
            for _ in range(batch_size):
                perm_idx = B.permutation(n)
                Y_perm = Y_gpu[perm_idx]


                sX = X_gpu.reshape(-1, 1) * self.freq_s_gpu.reshape(1, -1)
                tY = Y_perm.reshape(-1, 1) * self.freq_t_gpu.reshape(1, -1)

                phi_X_real = B.mean(B.cos(sX), axis=0)
                phi_X_imag = B.mean(B.sin(sX), axis=0)
                phi_Y_real = B.mean(B.cos(tY), axis=0)
                phi_Y_imag = B.mean(B.sin(tY), axis=0)

                arg_XY = sX + tY
                phi_XY_real = B.mean(B.cos(arg_XY), axis=0)
                phi_XY_imag = B.mean(B.sin(arg_XY), axis=0)


                delta_real = phi_XY_real - phi_X_real * phi_Y_real + phi_X_imag * phi_Y_imag
                delta_imag = phi_XY_imag - phi_X_real * phi_Y_imag - phi_X_imag * phi_Y_real
                delta_sq = delta_real ** 2 + delta_imag ** 2

                stat = n * B.mean(delta_sq)
                batch_stats.append(float(B.to_numpy(stat)))

            perm_statistics.extend(batch_stats)

        if hasattr(B, 'synchronize'):
            B.synchronize()


        perm_statistics = np.array(perm_statistics[:self.n_permutations])
        p_value = (np.sum(perm_statistics >= observed_statistic) + 1) / (self.n_permutations + 1)

        return float(p_value)

    def permutation_test_federated(
        self,
        X_list: List[np.ndarray],
        Y_list: List[np.ndarray],
        Z_list: Optional[List[np.ndarray]],
        observed_statistic: float
    ) -> float:


        if Z_list is not None:
            X_residuals = []
            Y_residuals = []
            for k in range(len(X_list)):
                X_res = self._compute_residuals_cpu(X_list[k], Z_list[k])
                Y_res = self._compute_residuals_cpu(Y_list[k], Z_list[k])
                X_residuals.append(X_res)
                Y_residuals.append(Y_res)
            X_all = np.vstack(X_residuals)
            Y_all = np.vstack(Y_residuals)
        else:
            X_all = np.vstack([self._ensure_2d(x) for x in X_list])
            Y_all = np.vstack([self._ensure_2d(y) for y in Y_list])


        return self._batch_permutation_test_gpu(X_all, Y_all, observed_statistic)

    def test(
        self,
        X_list: List[np.ndarray],
        Y_list: List[np.ndarray],
        Z_list: Optional[List[np.ndarray]] = None,
        verbose: bool = True
    ) -> Dict:
        n_clients = len(X_list)

        if verbose:
            test_type = "X ⊥⊥ Y | Z" if Z_list else "X ⊥⊥ Y"
            print(f"\n{'='*70}")
            print(f"🚀 FedCFCIT-GPU: Testing {test_type}")
            print(f"{'='*70}")
            print(f"   Backend: {self.backend.name}")
            print(f"   Device: {self.backend.device}")
            print(f"   Number of clients: {n_clients}")
            print(f"   Number of frequencies: {self.n_frequencies}")

        start_time = time.time()


        if verbose:
            print("\n⚙️  Computing local statistics...")

        local_stats_list = []
        for k in range(n_clients):
            Z_k = Z_list[k] if Z_list else None
            local_stats = self.compute_local_statistics(X_list[k], Y_list[k], Z_k)
            local_stats_list.append(local_stats)


        global_stats = self.aggregate_statistics(local_stats_list)


        statistic, cf_distance_squared = self.compute_test_statistic(global_stats)

        if verbose:
            print(f"   Statistic T = {statistic:.6f}")


        if verbose:
            print(f"🎲 Permutation test ({self.n_permutations} runs)...")

        if self.use_permutation:
            p_value = self.permutation_test_federated(
                X_list, Y_list, Z_list, statistic
            )
        else:

            p_value = 0.5

        elapsed = time.time() - start_time


        reject_null = p_value < self.alpha
        decision = 'dependent' if reject_null else 'independent'

        if verbose:
            print(f"\n📋 Results:")
            print(f"   p-value = {p_value:.4f}")
            print(f"   Decision: {decision}")
            print(f"   Elapsed: {elapsed:.3f} s")
            print(f"{'='*70}\n")

        return {
            'statistic': statistic,
            'cf_distance_squared': cf_distance_squared,
            'pvalue': p_value,
            'reject_null': reject_null,
            'decision': decision,
            'n_samples': global_stats['n_total'],
            'n_clients': n_clients,
            'elapsed_time': elapsed,
            'backend': self.backend.name,
            'device': self.backend.device
        }


def federated_cf_ci_test_gpu(
    X_list: List[np.ndarray],
    Y_list: List[np.ndarray],
    Z_list: Optional[List[np.ndarray]] = None,
    backend: str = 'auto',
    **kwargs
) -> Dict:
    tester = FederatedCFCI_GPU(backend=backend, **kwargs)
    return tester.test(X_list, Y_list, Z_list, verbose=False)


def benchmark_backends():
    print("=" * 70)
    print("FedCFCIT Backend Performance Benchmark")
    print("=" * 70)

    np.random.seed(42)


    test_configs = [
        {'n_clients': 5, 'n_samples': 500, 'name': 'Small'},
        {'n_clients': 5, 'n_samples': 2000, 'name': 'Medium'},
        {'n_clients': 10, 'n_samples': 5000, 'name': 'Large'},
    ]

    backends_to_test = ['numpy']


    try:
        import cupy
        backends_to_test.append('cupy')
    except ImportError:
        print("⚠️  CuPy not available")

    try:
        import torch
        if torch.cuda.is_available():
            backends_to_test.append('torch')
        else:
            print("⚠️  PyTorch CUDA not available")
    except ImportError:
        print("⚠️  PyTorch not available")

    print(f"\nAvailable backends: {backends_to_test}\n")

    results = []

    for config in test_configs:
        print(f"\n--- {config['name']} (n={config['n_samples']}, K={config['n_clients']}) ---")


        X_list = [np.random.randn(config['n_samples'], 1) for _ in range(config['n_clients'])]
        Y_list = [2 * X_list[k] + np.random.randn(config['n_samples'], 1) * 0.3
                  for k in range(config['n_clients'])]

        for backend in backends_to_test:
            try:
                tester = FederatedCFCI_GPU(
                    backend=backend,
                    n_frequencies=50,
                    n_permutations=100
                )


                _ = tester.test(X_list[:1], Y_list[:1], verbose=False)


                start = time.time()
                result = tester.test(X_list, Y_list, verbose=False)
                elapsed = time.time() - start

                print(f"  {backend:8s}: {elapsed:.3f}s, p={result['pvalue']:.4f}")

                results.append({
                    'config': config['name'],
                    'backend': backend,
                    'time': elapsed,
                    'pvalue': result['pvalue']
                })

            except Exception as e:
                print(f"  {backend:8s}: Error - {e}")


    if len(backends_to_test) > 1:
        print("\n--- Speedup (relative to NumPy) ---")
        for config in test_configs:
            numpy_time = next((r['time'] for r in results
                             if r['config'] == config['name'] and r['backend'] == 'numpy'), None)
            if numpy_time:
                for backend in backends_to_test[1:]:
                    other_time = next((r['time'] for r in results
                                      if r['config'] == config['name'] and r['backend'] == backend), None)
                    if other_time:
                        speedup = numpy_time / other_time
                        print(f"  {config['name']:8s} - {backend}: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("FedCFCIT GPU-accelerated version - Test")
    print("=" * 70)


    backend = get_backend('auto')
    print(f"\nAuto-detected backend: {backend.name}")
    print(f"Compute device: {backend.device}")


    np.random.seed(42)
    n_clients = 5
    n_samples = 500

    print(f"\nGenerating test data: {n_clients} clients, {n_samples} samples each")

    X_list = [np.random.randn(n_samples, 1) for _ in range(n_clients)]
    Y_list = [2 * X_list[k] + np.random.randn(n_samples, 1) * 0.3
              for k in range(n_clients)]


    result = federated_cf_ci_test_gpu(
        X_list, Y_list,
        backend='auto',
        n_frequencies=50,
        n_permutations=200
    )

    print("\nRunning benchmark...")
    benchmark_backends()
