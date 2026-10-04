# -*- coding: utf-8 -*-
"""临时 CSV 工程化（§18）：文件权限 / 规模与磁盘检查 / 清理登记。"""

from __future__ import annotations

import os
import shutil
import stat

import pytest

from backends.psi_backend import runtime as rt


class TestWriteKeys:
    def test_file_is_private_and_content_is_csv(self, tmp_path):
        path = str(tmp_path / "keys.csv")
        rt._write_keys(path, [2, 1])
        assert os.path.exists(path)
        if os.name == "posix":
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == "grid_code\n2\n1\n"

    def test_invalid_code_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="2\^64"):
            rt._write_keys(str(tmp_path / "keys.csv"), [2 ** 64])


class TestIoBudget:
    def test_estimate_is_conservative(self):
        assert rt._estimate_csv_bytes(0) == 64
        assert rt._estimate_csv_bytes(10) > 10 * 20

    def test_accepts_normal_sizes(self):
        assert rt._io_budget_problem(3, 3, 10 ** 9) is None

    def test_rejects_too_many_codes_with_advice(self, monkeypatch):
        monkeypatch.setattr(rt, "PSI_INPUT_MAX_CODES", 4)
        problem = rt._io_budget_problem(5, 1, 10 ** 9)
        assert problem is not None
        assert "超过单方上限" in problem
        assert "partition_grid_codes" in problem

    def test_rejects_insufficient_disk_space(self):
        problem = rt._io_budget_problem(1, 1, 0)
        assert problem is not None
        assert "剩余空间不足" in problem


class TestWorkdirLifecycle:
    def test_make_workdir_is_private_and_registered(self):
        path = rt._make_workdir()
        try:
            assert os.path.isdir(path)
            assert path in rt._ACTIVE_WORKDIRS
            if os.name == "posix":
                assert stat.S_IMODE(os.stat(path).st_mode) == 0o700
        finally:
            rt._release_workdir(path)
            shutil.rmtree(path, ignore_errors=True)
        assert path not in rt._ACTIVE_WORKDIRS

    def test_exit_cleanup_removes_registered_workdirs(self):
        path = rt._make_workdir()
        assert os.path.isdir(path)
        rt._cleanup_active_workdirs()
        assert not os.path.exists(path)
        assert not rt._ACTIVE_WORKDIRS
