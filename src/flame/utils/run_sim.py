import logging
import shutil
from pathlib import PosixPath
from typing import Callable

from .urg_parser import get_cov_item
from .utils import Utils


def run_sim(
    start_index: int,
    input_bins: list,
    output_dir: str | PosixPath,
    sim_path: str,
    get_sim_cmd: Callable[[int, PosixPath, PosixPath], dict],
    get_each_cov_result_path: Callable[[PosixPath], PosixPath | None],
    get_sim_log_path: Callable[[PosixPath], PosixPath],
    dest_sim_log_path: PosixPath | None = None,
    timeout_s: int = 86400,
) -> int:
    utils = Utils()
    logging.info(
        f"{start_index} batch simulation starts, input_bins: {input_bins}, output_dir: {output_dir}"
    )
    output_dir = PosixPath(output_dir)
    max_iter = len(input_bins)
    input_bins.sort(key=lambda x: int(PosixPath(x).stem))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_output_dir = output_dir / "output"
    out_output_dir.mkdir(parents=True, exist_ok=True)

    for dir in output_dir.iterdir():
        if dir.match("*temp_bins*") or dir.match("*temp_out*"):
            shutil.rmtree(dir)
    temp_bins_dir = output_dir / ("temp_bins_" + utils.get_time())
    temp_bins_dir.mkdir(parents=True, exist_ok=True)
    temp_out_dir = output_dir / ("temp_out_" + utils.get_time())
    if temp_out_dir.exists():
        shutil.rmtree(temp_out_dir)
    for bin_file in input_bins:
        index = input_bins.index(bin_file)
        temp_bins_dir.joinpath(str(index)).mkdir(parents=True, exist_ok=True)
        dest_bin_file = (
            temp_bins_dir / str(index) / (str(index) + PosixPath(bin_file).suffix)
        )
        shutil.copyfile(bin_file, dest_bin_file)
    sim_cmd_dict = get_sim_cmd(
        max_iter, temp_out_dir.absolute(), dest_bin_file.absolute()
    )
    sim_cmd = sim_cmd_dict["cmd"]
    sim_cmd_env = sim_cmd_dict["env"]
    sim_result, _ = utils.run_cmd(
        cmd=sim_cmd, cwd=sim_path, timeout_s=timeout_s, env=sim_cmd_env
    )
    logging.info("Simulation result (run_sim): %s" % sim_result)
    if temp_bins_dir.exists():
        shutil.rmtree(temp_bins_dir)

    if sim_result == utils.RET_FAIL:
        if temp_out_dir.exists():
            shutil.rmtree(temp_out_dir)
        return utils.RET_FAIL
    elif sim_result == utils.RET_TIMEOUT:
        if temp_out_dir.exists():
            shutil.rmtree(temp_out_dir)
        return utils.RET_TIMEOUT

    each_sim_log_path = get_sim_log_path(temp_out_dir.absolute())
    each_sim_log_path = PosixPath(each_sim_log_path)
    if dest_sim_log_path:
        if not each_sim_log_path.exists():
            logging.error(f"[SIM] Sim log {each_sim_log_path} not found")
            each_sim_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(each_sim_log_path, "w") as f:
                f.write(f"[ERROR] Sim log not found: {PosixPath(each_sim_log_path).absolute()}")
        shutil.copyfile(each_sim_log_path, dest_sim_log_path)

    for i in range(max_iter):
        each_cov_result_path = get_each_cov_result_path(temp_out_dir.absolute())
        dest_cov_result_path = out_output_dir / str(start_index * max_iter + i + 1)

        if (each_cov_result_path is None) or (not each_cov_result_path.exists()):
            shutil.rmtree(temp_out_dir)
            return utils.RET_FAIL

        if dest_cov_result_path.exists():
            shutil.rmtree(dest_cov_result_path)
        shutil.copytree(each_cov_result_path, dest_cov_result_path)
        cov_result_file = dest_cov_result_path / "grpinfo.txt"
        dest_cov_result_item_file = dest_cov_result_path / "cov.txt"
        cov_item = get_cov_item(cov_result_file)["covered"]
        cov_item.sort()
        with dest_cov_result_item_file.open("w") as f:
            f.writelines([f"{_}\n" for _ in cov_item])
        logging.info(f"Coverage saved to {dest_cov_result_path}")
    if temp_out_dir.exists():
        shutil.rmtree(temp_out_dir)
    logging.info("%d batch simulation finished" % start_index)
    return 0
