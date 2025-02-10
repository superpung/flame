from pathlib import PosixPath
from flame.utils import ChatBot, Utils

from .base_runner import RunX


class RunIbexV1(RunX):
    def __init__(
        self,
        output: str | PosixPath,
        bot: ChatBot | None = None,
        iter: int = 100,
        prompt_type: str | None = "cot",
        input_type: str | None = "c",
        options: str | None = None,
    ):
        super().__init__(dut="ibexv1", output=output, bot=bot, iter=iter, prompt_type=prompt_type, input_type=input_type, options=options)
        self.sim_timeout_s = 180
        self.sim_path = "dut/ibex_v1/dv/uvm/core_ibex"

    def compile(self, src_file):
        return Utils.compile_ibexv1(c_file=src_file, sim_path=self.sim_path)

    def get_c_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        sim_path = PosixPath(self.sim_path).absolute()
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = f"make -C {sim_path} SIMULATOR=vcs ISA=rv32imc_zicsr_zifencei ISS=spike TEST=riscv_custom_test ITERATIONS=1 COV=1 OUT={out_dir}/out INPUT_DIR={input_dir} OUT-SEED={out_dir}"
        res = {"cmd": sim_cmd, "env": {}}
        return res

    def get_asm_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        sim_path = PosixPath(self.sim_path).absolute()
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = f"make -C {sim_path} SIMULATOR=vcs ISA=rv32imc_zicsr_zifencei ISS=spike TEST=riscv_custom_test ITERATIONS=1 COV=1 OUT={out_dir}/out INPUT_DIR={input_dir} OUT-SEED={out_dir}"
        res = {"cmd": sim_cmd, "env": {}}
        return res

    def get_each_cov_result_path(self, out_dir: PosixPath) -> PosixPath | None:
        return out_dir / "rtl_sim/coverage/1"

    def get_sim_log_path(self, out_dir: PosixPath) -> PosixPath:
        return out_dir / "rtl_sim/sim_logs/sim.log"
