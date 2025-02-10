from pathlib import PosixPath

from flame.utils import ChatBot, Utils

from .base_runner import RunX


class RunCV32E40P(RunX):
    def __init__(
        self,
        output: str | PosixPath,
        bot: ChatBot | None = None,
        iter: int = 100,
        prompt_type: str | None = "cot",
        input_type: str | None = "c",
        options: str | None = None,
    ):
        super().__init__(dut="cv32e40p", output=output, bot=bot, iter=iter, prompt_type=prompt_type, input_type=input_type, options=options)
        self.sim_timeout_s = 180
        self.sim_path = "dut/core-v-verif/cv32e40p/sim/uvmt"

    def compile(self, src_file):
        return Utils.compile_cv32e40p(c_file=src_file, sim_path=self.sim_path)

    def get_c_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        sim_path = PosixPath(self.sim_path).absolute()
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = f"make -C {sim_path} test TEST=test TEST_PATH={input_dir} SIMULATOR=vcs COV=1 USE_ISS=NO SIM_RESULTS={out_dir}"
        res = {"cmd": sim_cmd, "env": {}}
        return res

    def get_asm_sim_cmd(self, max_iter, out_dir: str | PosixPath, input_dir: str | PosixPath) -> dict:
        sim_path = PosixPath(self.sim_path).absolute()
        out_dir = PosixPath(out_dir).absolute()
        input_dir = PosixPath(input_dir).absolute()
        sim_cmd = f"make -C {sim_path} test TEST=test TEST_PATH={input_dir} SIMULATOR=vcs COV=1 USE_ISS=NO SIM_RESULTS={out_dir}"
        res = {"cmd": sim_cmd, "env": {}}
        return res

    def get_each_cov_result_path(self, out_dir: PosixPath) -> PosixPath | None:
        return out_dir / "default/coverage"

    def get_sim_log_path(self, out_dir: PosixPath) -> PosixPath:
        return out_dir / "default/test/0/vcs-test.log"
