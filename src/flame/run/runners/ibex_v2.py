from pathlib import PosixPath
from flame.utils import ChatBot, Utils

from .base_runner import RunX


class RunIbexV2(RunX):
    def __init__(
        self,
        output: str | PosixPath,
        bot: ChatBot | None = None,
        iter: int = 100,
        prompt_type: str | None = "cot",
        input_type: str | None = "c",
        options: str | None = None,
    ):
        super().__init__(dut="ibexv2", output=output, bot=bot, iter=iter, prompt_type=prompt_type, input_type=input_type, options=options)
        self.sim_timeout_s = 2400
        self.sim_path = "dut/ibex_v2/dv/uvm/core_ibex"

    def compile(self, src_file):
        return Utils.compile_ibexv2(c_file=src_file, sim_path=self.sim_path)

    def get_c_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        sim_path = PosixPath(self.sim_path).absolute()
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = f"make -C {sim_path} SIMULATOR=vcs ISA=rv32imc_zicsr_zifencei ISS=spike TEST=riscv_custom_test ITERATIONS=1 COV=1 GOAL=rtl_sim_run INPUT_DIR={input_dir} OUT={out_dir}"
        res = {"cmd": sim_cmd, "env": {}}
        return res

    def get_asm_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        sim_path = PosixPath(self.sim_path).absolute()
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = f"make -C {sim_path} SIMULATOR=vcs ISA=rv32imc_zicsr_zifencei ISS=spike TEST=riscv_custom_test ITERATIONS=1 COV=1 GOAL=rtl_sim_run INPUT_DIR={input_dir} OUT-SEED={out_dir}"
        res = {"cmd": sim_cmd, "env": {}}
        return res

    def get_each_cov_result_path(self, out_dir: PosixPath) -> PosixPath | None:
        test_dir = out_dir / "run/coverage/urgReport"
        if not test_dir.exists() or not list(test_dir.iterdir()):
            return None
        sub_test_dir = list(test_dir.iterdir())[0]
        return sub_test_dir

    def get_sim_log_path(self, out_dir: PosixPath) -> PosixPath:
        test_dir = out_dir / "run/tests"
        sub_test_dir = list(test_dir.iterdir())[0]
        return sub_test_dir / "rtl_sim_stdstreams.log"
