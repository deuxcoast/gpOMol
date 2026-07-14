"""
wl_gp2scale
===========
Explicit-vocabulary Weisfeiler-Lehman descriptor + distributed, block-sparse
gp2Scale GP kernel for scaling Gaussian-process energy regression to 200k
molecules on 16 GPUs (Perlmutter).

Self-contained: imports nothing from descriptor_eval/ or hybrid_descriptor/.

Public API
----------
    data.get_data / data.Dataset / data.stratified_sample_indices
    wl_features.SparseWLFeaturizer          -> scipy.sparse.csr_matrix
    reduce.SparsePLS                        streaming SIMPLS (sparse, supervised)
    kernel.make_wl_block_kernel             gp2Scale GPU block Wendland kernel
    kernel.check_kernel_psd                 PD falsification guard
    cutoff.recalibrate / cutoff.sparsity_report
    pipeline.WLGPPipeline / build_gp / predict / connect_dask
    validate.*                              pre-run checklist
"""

from . import cutoff, data, kernel, pipeline, reduce, wl_features  # noqa: F401
from .cutoff import recalibrate, sparsity_report  # noqa: F401
from .data import Dataset, get_data, stratified_sample_indices  # noqa: F401
from .kernel import check_kernel_psd, make_wl_block_kernel  # noqa: F401
from .pipeline import WLGPPipeline, build_gp, connect_dask, predict  # noqa: F401
from .reduce import SparsePLS  # noqa: F401
from .wl_features import SparseWLFeaturizer  # noqa: F401

__all__ = [
    "data", "wl_features", "reduce", "kernel", "cutoff", "pipeline",
    "get_data", "Dataset", "stratified_sample_indices",
    "SparseWLFeaturizer", "SparsePLS", "make_wl_block_kernel",
    "check_kernel_psd", "recalibrate", "sparsity_report",
    "WLGPPipeline", "build_gp", "predict", "connect_dask",
]
