from solvers.linear_types import JacobianMode


def test_jacobian_mode_normalize_basic():
    assert JacobianMode.normalize("fd") == JacobianMode.FD
    assert JacobianMode.normalize("MF") == JacobianMode.MF
    assert JacobianMode.normalize(None) == JacobianMode.FD
    assert JacobianMode.normalize("") == JacobianMode.FD


def test_jacobian_mode_normalize_aliases():
    assert JacobianMode.normalize("mfpc_sparse") == JacobianMode.MFPC_SPARSE_FD
    assert JacobianMode.normalize("mfpc_sparse_fd") == JacobianMode.MFPC_SPARSE_FD
    assert JacobianMode.normalize("mfpc_aij") == JacobianMode.MFPC_AIJA
    assert JacobianMode.normalize("mfpc_aija") == JacobianMode.MFPC_AIJA


def test_jacobian_mode_normalize_pass_through_unknown():
    assert JacobianMode.normalize("custom_mode") == "custom_mode"
