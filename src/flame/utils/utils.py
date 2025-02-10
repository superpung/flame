import logging
import os
import subprocess
from pathlib import PosixPath
from time import localtime, strftime, time

from dotenv import dotenv_values
from openai import OpenAI


class Config:
    def __init__(self):
        self.config = dotenv_values(
            PosixPath(__file__).parent.parent.parent.parent / ".env"
        )
        self.config_checker()

    def config_checker(self):
        assert all(
            key in self.config
            for key in [
                "LOCAL_API_KEY",
                "LOCAL_API_BASE",
                "PHIND_34B_MODEL",
                "CODELLAMA_13B_MODEL",
                "CODELLAMA_7B_MODEL",
                "GPT_35_TURBO_0125_MODEL",
            ]
        ), "Missing required keys in .env file"


class EmptyResponseError(Exception):
    pass


def chat(prompt, temperature, model):
    client = OpenAI(
        api_key="EMPTY",
        base_url="",
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )
    return response


class ChatBot:
    def __init__(self, model, api_key: str | None = None, base_url: str | None = None, temperature: float | None = None):
        self.config = Config().config
        self.client = OpenAI(
            api_key=self.config["LOCAL_API_KEY"] if not api_key else api_key,
            base_url=self.config["LOCAL_API_BASE"] if not base_url else base_url,
        )
        self.load_model(model)
        self.temperature = temperature

    def load_model(self, model):
        model = model.lower()
        if "phind" in model and "34b" in model:
            self.model = self.config["PHIND_34B_MODEL"]
        elif "codellama" in model and "13b" in model:
            self.model = self.config["CODELLAMA_13B_MODEL"]
        elif "codellama" in model and "7b" in model:
            self.model = self.config["CODELLAMA_7B_MODEL"]
        elif "gpt" in model and "3.5-turbo-0125" in model:
            self.config = Config().config
            self.client = OpenAI(
                api_key=self.config["OPENAI_API_KEY"],
                base_url=self.config["OPENAI_API_BASE"],
            )
            self.model = self.config["GPT_35_TURBO_0125_MODEL"]
        else:
            raise ValueError(f"Model {model} not found")

    def chat(self, prompt: str = "", system_prompt="", messages=[], temperature: float | None = None) -> str:
        if self.temperature is None and temperature is None:
            temperature = 0.7
        elif temperature is None:
            temperature = self.temperature
        if messages != []:
            logging.debug("MESSAGES:\n%s" % messages)
            prompt_messages = [
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in messages
            ]
        else:
            logging.debug("SYSTEM PROMPT:\n%s" % system_prompt)
            logging.debug("PROMPT:\n%s" % prompt)
            prompt_messages = [
                {"role": "user", "content": prompt},
            ]
        response = self.client.chat.completions.create(
            messages=prompt_messages,  # type: ignore
            model=self.model,  # type: ignore
            temperature=temperature,
        )
        response = response.choices[0].message.content  # type: ignore
        if not response:
            raise EmptyResponseError("Empty response")
        logging.debug("RESPONSE: %s" % response)
        return response


class Utils:
    RET_FAIL = 1
    RET_TIMEOUT = 2

    DISABLE_OPTIONS_MAP = {
        "document": "-doc",
        "definition": "-def",
        "isa": "-isa",
        "counterexamples": "-ce",
        "feedback": "-fb",
    }

    @staticmethod
    def valid_url(url):
        import re
        return not not re.match(r"^https?://\w.+$", url)

    @staticmethod
    def get_time():
        return strftime("%y%m%d%H%M%S", localtime(time()))

    @staticmethod
    def save_code(
        response, path: str | PosixPath | None = None, format: str = "c"
    ) -> str:
        format = format.lower()
        path = PosixPath(path) if path else None
        code_in_completions = response.split("```")
        if len(code_in_completions) < 2:
            code_in_completion = code_in_completions[0]
        else:
            code_in_completion = code_in_completions[1]
        if format == "c":
            lang = ["c", "C", "c++", "C++"]
        else:
            lang = ["asm", "ASM", "assembly", "Assembly"]
        for l in lang:  # noqa: E741
            if code_in_completion.startswith(l):
                code_in_completion = code_in_completion[len(l) :]
                break
        if path is not None:
            with path.open("w", encoding="utf-8") as f:
                f.write(code_in_completion)
        logging.info(f"Code saved to {path}")
        return code_in_completion

    @staticmethod
    def record_time(output_dir, label, action="default"):
        output_dir = PosixPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "log.txt"
        current_time = int(time())
        with log_file.open("a") as f:
            f.write(f"{label},{action},{current_time}\n")
        return current_time

    @staticmethod
    def get_time_used_from_log(log_file, stop_n_contain_iter):
        log_file = PosixPath(log_file)
        time_used = 0
        for line in log_file.read_text().strip().split("\n"):
            splited = line.split(",")
            if len(splited) != 3:
                continue
            if int(splited[0]) > stop_n_contain_iter:
                break
            timestamp = int(splited[-1])
            if "finish sim" in line:
                time_used += timestamp - start_sim_time
            elif "start sim" in line:
                start_sim_time = timestamp
            elif "start gen" in line:
                start_gen_time = timestamp
            elif "finish gen" in line:
                time_used += timestamp - start_gen_time
        return time_used

    @staticmethod
    def get_time_dict_from_dirs(out_dirs):
        result = []
        for out_dir in out_dirs:
            result.append(Utils.get_time_dict(out_dir))
        return result

    @staticmethod
    def get_time_dict(out_dir):
        out_dir = PosixPath(out_dir)
        time_file = out_dir.joinpath("log.txt")
        if not time_file.exists():
            return {}
        index_set = set()
        times = {}
        for timestamp in time_file.read_text().strip().split("\n"):
            input_index = int(timestamp.split(",")[0])
            cov_file = out_dir.joinpath(f"output/{input_index + 1}/cov.txt")
            before_history_cov_file = out_dir.joinpath(f"output/{input_index}/cov_up_to_now.txt")
            now_history_cov_file = out_dir.joinpath(f"output/{input_index + 1}/cov_up_to_now.txt")
            if not cov_file.exists():
                continue
            index_set.add(input_index)
            cov_up_to_now = len(set(now_history_cov_file.read_text().strip().split("\n")))
            if before_history_cov_file.exists():
                more_cov = set(cov_file.read_text().strip().split("\n")) - set(before_history_cov_file.read_text().strip().split("\n"))
            else:
                more_cov = set(cov_file.read_text().strip().split("\n"))
            if "start sim" in timestamp:
                start_sim = float(timestamp.split(",")[-1])
            elif "finish sim" in timestamp:
                finish_sim = float(timestamp.split(",")[-1])
                if input_index not in times:
                    times[input_index] = {"sim_time": [finish_sim - start_sim], "more_cov": len(more_cov), "current_cov": cov_up_to_now}
                elif "sim_time" not in times[input_index]:
                    times[input_index]["sim_time"] = [finish_sim - start_sim]
                else:
                    times[input_index]["sim_time"].append(finish_sim - start_sim)
            elif "start gen" in timestamp:
                start_gen = float(timestamp.split(",")[-1])
            elif "finish gen" in timestamp:
                finish_gen = float(timestamp.split(",")[-1])
                if input_index not in times:
                    times[input_index] = {"gen_time": [finish_gen - start_gen], "more_cov": len(more_cov), "current_cov": cov_up_to_now}
                elif "gen_time" not in times[input_index]:
                    times[input_index]["gen_time"] = [finish_gen - start_gen]
                else:
                    times[input_index]["gen_time"].append(finish_gen - start_gen)
        return times

    @staticmethod
    def cov_to_cov_time(out_dirs):
        for out_dir in out_dirs:
            print("...converting cov.csv to cov_time.csv in", out_dir)
            out_dir = PosixPath(out_dir)
            cov_time_file = out_dir / "cov_time.csv"
            if cov_time_file.exists():
                cov_time_file.unlink()
            cov_time_dict = {}
            cov_file = out_dir / "cov.csv"
            cov_dict = {int(_.split(",")[0]): int(_.split(",")[1]) for _ in cov_file.read_text().strip().split("\n")}
            time_used = 0
            time_dict = Utils.get_time_dict(out_dir=out_dir)
            for iter, times in time_dict.items():
                sim_time = sum(times["sim_time"]) if "sim_time" in times else 0
                gen_time = sum(times["gen_time"]) if "gen_time" in times else 0
                time_used += sim_time + gen_time
                cov_time_dict[int(time_used)] = cov_dict[iter + 1]
            with cov_time_file.open("w") as f:
                for time, cov in cov_time_dict.items():
                    f.write(f"{time},{cov}\n")

    @staticmethod
    def check_sim_gen_time(out_dirs, max_iter=1000, debug: bool = False):
        result = {}
        for label_out_dir in out_dirs:
            label = label_out_dir[0]
            out_dir = label_out_dir[1]
            if debug:
                print("...", out_dir)
            out_dir = PosixPath(out_dir)
            times = Utils.get_time_dict(out_dir=out_dir)
            label += f" {len({k: v for k, v in times.items() if k <= max_iter})}"
            result[label] = {
                "Success Sim.": sum([v["sim_time"][-1] if "sim_time" in v else 0 for k, v in times.items() if k <= max_iter]),
                "Error Sim.": sum([sum(v["sim_time"][:-1]) if "sim_time" in v else 0 for k, v in times.items() if k <= max_iter]),
                "Success Gen.": sum([v["gen_time"][-1] if "gen_time" in v else 0 for k, v in times.items() if k <= max_iter]),
                "Error Gen.": sum([sum(v["gen_time"][:-1]) if "gen_time" in v else 0 for k, v in times.items() if k <= max_iter]),
            }
            result[label] = {k: v / 3600 for k, v in result[label].items()}
        return result


    @staticmethod
    def run_cmd(
        cmd,
        cwd=None,
        timeout_s=3000,
        exit_on_error=1,
        check_return_code=True,
        env=None,
    ) -> tuple:
        logging.info(cmd)
        try:
            os_env = os.environ.copy()
            os_env.update(env or {})
            ps = subprocess.Popen(
                "exec " + cmd,
                shell=True,
                executable="/bin/bash",
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=os_env,
            )
        except subprocess.CalledProcessError:
            output = ps.communicate()[0]  # type: ignore
            logging.error(output)
            return Utils.RET_FAIL, output
        except KeyboardInterrupt:
            output = "\nExited Ctrl-C from user request."
            logging.info(output)
            return Utils.RET_FAIL, output
        try:
            output = ps.communicate(timeout=timeout_s)[0]
        except subprocess.TimeoutExpired:
            output = "Timeout[{}s]: {}".format(timeout_s, cmd)
            ps.kill()
            logging.error(output)
            return Utils.RET_TIMEOUT, output
        except UnicodeDecodeError as e:
            logging.error(e)
            return Utils.RET_FAIL, e
        rc = ps.returncode
        if rc and check_return_code and rc > 0:
            logging.info(output)
            logging.error(
                "ERROR return code: {}/{}, cmd:{}".format(check_return_code, rc, cmd)
            )
            if exit_on_error:
                return Utils.RET_FAIL, output
        logging.debug(output)
        return 0, output

    @staticmethod
    def run_cmds(
        cmds: list,
        cwd=None,
        timeout_s=3000,
        exit_on_error=1,
        check_return_code=True,
        env=None,
    ) -> tuple:
        output_list = []
        for cmd in cmds:
            result, output = Utils.run_cmd(cmd, cwd, timeout_s, exit_on_error, check_return_code, env)
            if result in [Utils.RET_FAIL, Utils.RET_TIMEOUT]:
                return result, output
            output_list.append(output)
        return 0, output_list

    @staticmethod
    def compile_ibexv1(
        c_file: str | PosixPath,
        sim_path: str | PosixPath | None = None,
        dest_bin_file: str | PosixPath | None = None,
        contains_asm: bool = False,
    ) -> tuple:
        c_file = PosixPath(c_file)
        output_file = c_file.parent / (c_file.stem + ".o")
        output_bin_file = c_file.parent / (c_file.stem + ".bin")
        if c_file.suffix in [".s", ".S"]:
            # assembly code
            compile_o_cmd = (
                f"{os.environ['RISCV_GCC']} "
                "-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -mno-strict-align "
                f"{c_file.absolute()} "
                "-I../../../vendor/google_riscv-dv/user_extension "
                "-T../../../vendor/google_riscv-dv/scripts/link.ld "
                f"-o {output_file.absolute()} "
                "-march=rv32imc_zicsr_zifencei -mabi=ilp32"
            )
            sim_result, sim_output = Utils.run_cmd(cmd=compile_o_cmd, cwd=sim_path)
            compile_asm_cmd = f"{os.environ['RISCV_OBJCOPY']} -O binary {output_file.absolute()} {output_bin_file.absolute()}"
            sim_result, sim_output = Utils.run_cmd(cmd=compile_asm_cmd, cwd=sim_path)
            return sim_result, sim_output
        else:
            # C program
            cmd = (
                f"{os.environ['RISCV_GCC']} "
                "-mcmodel=medany -nostartfiles "
                f"{str(c_file.absolute())} "
                "-I../../../vendor/google_riscv-dv/user_extension "
                "-T../../../vendor/google_riscv-dv/scripts/link.ld "
                f"-o {str(output_file.absolute())} "
                "-march=rv32imc_zicsr_zifencei -mabi=ilp32"
            )
        sim_result, sim_output = Utils.run_cmd(cmd=cmd, cwd=sim_path)
        return sim_result, sim_output

    @staticmethod
    def compile_ibexv2(
        c_file: str | PosixPath,
        sim_path: str | PosixPath | None = None,
        dest_bin_file: str | PosixPath | None = None,
        contains_asm: bool = False,
    ) -> tuple:
        c_file = PosixPath(c_file)
        output_file = c_file.parent / (c_file.stem + ".o")
        output_bin_file = c_file.parent / (c_file.stem + ".bin")
        if c_file.suffix in [".s", ".S"]:
            # assembly code
            compile_o_cmd = (
                f"{os.environ['RISCV_GCC']} "
                "-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -mno-strict-align "
                f"{c_file.absolute()} "
                "-I../../../vendor/google_riscv-dv/user_extension "
                "-T../../../vendor/google_riscv-dv/scripts/link.ld "
                f"-o {output_file.absolute()} "
                "-march=rv32imc_zicsr_zifencei -mabi=ilp32"
            )
            sim_result, sim_output = Utils.run_cmd(cmd=compile_o_cmd, cwd=sim_path)
            compile_asm_cmd = f"{os.environ['RISCV_OBJCOPY']} -O binary {output_file.absolute()} {output_bin_file.absolute()}"
            sim_result, sim_output = Utils.run_cmd(cmd=compile_asm_cmd, cwd=sim_path)
            return sim_result, sim_output
        else:
            # C program
            cmd = (
                f"{os.environ['RISCV_GCC']} "
                "-mcmodel=medany -nostartfiles "
                f"{str(c_file.absolute())} "
                "-I../../../vendor/google_riscv-dv/user_extension "
                "-T../../../vendor/google_riscv-dv/scripts/link.ld "
                f"-o {str(output_file.absolute())} "
                "-mno-strict-align "
                "-march=rv32imc_zicsr_zifencei -mabi=ilp32"
            )
        sim_result, sim_output = Utils.run_cmd(cmd=cmd, cwd=sim_path)
        return sim_result, sim_output

    @staticmethod
    def compile_cva6(
        c_file: str | PosixPath,
        sim_path: str,
        dest_bin_file: str | PosixPath | None = None,
        contains_asm: bool = False,
    ) -> tuple:
        c_file = PosixPath(c_file)
        output_file = c_file.parent / (c_file.stem + ".o")
        if c_file.suffix in [".s", ".S"]:
            # assembly code
            cmd = (
                f"{os.environ['RISCV_GCC']} "
                f"{str(c_file.absolute())} "
                "-I../env/corev-dv/user_extension "
                "-T./link.ld -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles "
                "-I../tests/custom/env -I../tests/custom/common "
                f"-o {str(output_file.absolute())} "
                "-march=rv32imc_zba_zbb_zbs_zbc_zicsr_zifencei -mabi=ilp32"
            )
        else:
            # C program
            cmd = (
                f"{os.environ['RISCV_GCC']} "
                f"{str(c_file.absolute())} "
                "-I./dv/user_extension "
                "-T../tests/custom/common/test.ld -static -mcmodel=medany -fvisibility=hidden -nostdlib "
                "-nostartfiles -g ../tests/custom/common/syscalls.c ../tests/custom/common/crt.S -lgcc "
                "-I../tests/custom/env -I../tests/custom/common "
                f"-o {str(output_file.absolute())} "
                "-march=rv32imac_zba_zbb_zbs_zbc_zicsr_zifencei -mabi=ilp32"
            )
        sim_result, sim_output = Utils.run_cmd(cmd=cmd, cwd=sim_path)
        return sim_result, sim_output

    @staticmethod
    def compile_cv32e40p(
        c_file: str | PosixPath,
        sim_path: str | PosixPath,
        dest_bin_file: str | PosixPath | None = None,
        contains_asm: bool = False,
    ) -> tuple:
        c_file = PosixPath(c_file)
        output_file = c_file.parent / (c_file.stem + ".elf")
        sim_path = PosixPath(sim_path).absolute()
        cmd = (
            f"{os.environ['RISCV_GCC']} "
            "-DNO_PULP "
            "-Os -g -static -mabi=ilp32 -march=rv32imc_zicsr_zifencei -Wall -pedantic  "
            f"-I {sim_path}/../../tests/asm "
            f"-o {output_file.absolute()} "
            "-nostartfiles "
            f"{c_file.absolute()} "
            f"-T {sim_path}/../../bsp/link.ld"
        )
        sim_result, sim_output = Utils.run_cmd(cmd=cmd, cwd=sim_path)
        return sim_result, sim_output

    @staticmethod
    def is_cross_point(s: str) -> bool:
        return "cross" in s or "pmp_wr_exec_region" in s

    @staticmethod
    def get_coverpoint_name(s: str) -> str:
        if Utils.is_cross_point(s):
            return s.split(".")[0]
        else:
            if "." not in s:
                return s
            return ".".join(s.split(".")[:-1])

    @staticmethod
    def get_cover_bin_name(s: str) -> str:
        if "." not in s:
            return s
        return s.split(".")[-1]

    @staticmethod
    def generate_c(
        bot: ChatBot,
        prompt: str,
        dest_path: str | PosixPath,
        max_retry: int = 10,
        contains_asm: bool = False,
    ) -> str:
        dest_path = PosixPath(dest_path)
        for _ in range(max_retry):
            response = bot.chat(prompt)
            code = Utils.save_code(response, dest_path)
            compile_result, _ = Utils.compile_ibexv1(
                c_file=dest_path.as_posix(),
                contains_asm=contains_asm,
            )
            if compile_result == 0:
                break
        return code

    @staticmethod
    def _s2h(s: int) -> str:
        h = int(s // 3600)
        s %= 3600
        m = int(s // 60)
        s %= 60
        s = int(s)
        res = f"{h}h {m}m {s}s"
        return res

    @staticmethod
    def calculate_time(output: str | PosixPath):
        output = PosixPath(output)
        log_file = output / "log.txt"
        res = {}
        times_info = [_.split(",") for _ in log_file.read_text().strip().split("\n")]
        total_time = int(times_info[-1][-1]) - int(times_info[0][-1])
        res["total_time"] = str(total_time) + f"s ({Utils._s2h(total_time)})"
        return res

    @staticmethod
    def get_instr_code_from_riscvdv(riscvdv_asm: str | PosixPath) -> str:
        logging.info(f"Get instruction code from {riscvdv_asm}")
        riscvdv_asm = PosixPath(riscvdv_asm)
        if not riscvdv_asm.exists():
            return ""
        with riscvdv_asm.open("r") as f:
            content = f.read()
        instruction_codes = content.split("mmode_intr_vector")
        if len(instruction_codes) < 2:
            content = Utils.filter_asm_instructions(riscvdv_asm)
            return content
        content = instruction_codes[1].strip().replace("_1:\n", "")
        return content

    @staticmethod
    def filter_asm_instructions(asm_file: str | PosixPath) -> str:
        asm_file = PosixPath(asm_file)
        if not asm_file.exists():
            return ""
        with asm_file.open("r") as f:
            contents = f.readlines()
        instructions = [
            _.strip()
            for _ in contents
            if not _.strip().startswith(".") and ":" not in _
        ]
        return "\n".join(instructions)
