from pathlib import PosixPath

from flame.utils import ChatBot, Utils

from .base_runner import RunX


class RunCVA6(RunX):
    def __init__(
        self,
        output: str | PosixPath,
        bot: ChatBot | None = None,
        iter: int = 100,
        prompt_type: str | None = "cot",
        input_type: str | None = "c",
        options: str | None = None,
    ):
        super().__init__(dut="cva6", output=output, bot=bot, iter=iter, prompt_type=prompt_type, input_type=input_type, options=options)
        self.sim_timeout_s = 600
        self.sim_path = "dut/cva6/verif/sim"

    def compile(self, src_file):
        return Utils.compile_cva6(c_file=src_file, sim_path=self.sim_path)

    def get_c_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = (
            f"python3 cva6.py --output={out_dir} --target=cv32a60x --iss=vcs-uvm --iss_yaml=cva6.yaml "
            f"--c_tests {input_dir} --verbose --linker=../tests/custom/common/test.ld "
            '--gcc_opts="-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -g ../tests/custom/common/syscalls.c ../tests/custom/common/crt.S -lgcc -I../tests/custom/env -I../tests/custom/common"'
        )
        res = {"cmd": sim_cmd, "env": {"VCS_WORK_DIR": out_dir / "vcs_results"}}
        return res

    def get_asm_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = (
            f"python3 cva6.py --output={out_dir} --target=cv32a60x --iss=vcs-uvm --iss_yaml=cva6.yaml "
            f"--asm_tests {input_dir} --verbose "
            '--gcc_opts="-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles"'
        )
        res = {"cmd": sim_cmd, "env": {"VCS_WORK_DIR": out_dir / "vcs_results"}}
        return res

    def get_each_cov_result_path(self, out_dir: PosixPath) -> PosixPath | None:
        return out_dir / "vcs_results/coverage"

    def get_sim_log_path(self, out_dir: PosixPath) -> PosixPath:
        return out_dir / "vcs-uvm_sim/0.cv32a60x.log.iss"
