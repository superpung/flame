from flame.utils import ChatBot, Config, Utils

from .runners import (
    RunCVA6,
    RunCV32E40P,
    RunIbexV1,
    RunIbexV2,
)

config = Config().config


class Runner:
    def __init__(self, args) -> None:
        print(args)
        self.method = args.method.lower()
        self.prompt = args.prompt.lower() if args.prompt else None
        self.dataset = args.dataset.lower()
        self.model = args.model.lower() if args.model else None
        self.iter = args.iter if args.iter else 100
        self.utils = Utils()
        self.output = args.output if args.output else self.utils.get_time()
        if not args.base_url:
            base_url = None
        elif Utils.valid_url(args.base_url):
            base_url = args.base_url
        else:
            raise ValueError(f"invalid base_url: {args.base_url}")
        self.base_url = base_url
        self.crv_start = args.crv_start
        self.options = args.options
        self.temperature = args.temperature

    def run(self):
        if "ibex" in self.dataset:
            if "v1" in self.dataset:
                self.run_ibex_v1()
            elif "v2" in self.dataset:
                self.run_ibex_v2()
        elif "cva6" in self.dataset:
            self.run_cva6()
        elif "cv32e40p" in self.dataset:
            self.run_cv32e40p()

    def run_ibex_v1(self):
        if "llm" in self.method:
            crv_start = self.crv_start.split(",") if self.crv_start else None
            output_dir = f"out/ibex_v1_{self.model}/{self.output}_{self.prompt}"
            output_dir += "_crv" if crv_start else ""
            bot = ChatBot(model=self.model, base_url=self.base_url, temperature=self.temperature)
            run_ibex_v1 = RunIbexV1(
                bot=bot,
                output=output_dir,
                iter=self.iter,
                prompt_type=self.prompt,
                options=self.options,
            )
            run_ibex_v1.run(crv_start=crv_start)
        else:
            raise ValueError(f"unknown method: {self.method}")

    def run_ibex_v2(self):
        if "llm" in self.method:
            crv_start = self.crv_start.split(",") if self.crv_start else None
            output_dir = f"out/ibex_v2_{self.model}/{self.output}_{self.prompt}"
            output_dir += "_crv" if crv_start else ""
            bot = ChatBot(model=self.model, base_url=self.base_url, temperature=self.temperature)
            run_ibex_v2 = RunIbexV2(
                bot=bot,
                output=output_dir,
                iter=self.iter,
                prompt_type=self.prompt,
                options=self.options,
            )
            run_ibex_v2.run(crv_start=crv_start)
        else:
            raise ValueError(f"unknown method: {self.method}")

    def run_cva6(self):
        if "llm" in self.method:
            crv_start = self.crv_start.split(",") if self.crv_start else None
            output_dir = f"out/cva6_{self.model}/{self.output}_{self.prompt}"
            output_dir += "_crv" if crv_start else ""
            bot = ChatBot(model=self.model, base_url=self.base_url, temperature=self.temperature)
            run_cva6 = RunCVA6(
                bot=bot,
                output=output_dir,
                iter=self.iter,
                prompt_type=self.prompt,
                options=self.options,
            )
            run_cva6.run(crv_start=crv_start)
        else:
            raise ValueError(f"unknown method: {self.method}")

    def run_cv32e40p(self):
        if "llm" in self.method:
            crv_start = self.crv_start.split(",") if self.crv_start else None
            output_dir = f"out/cv32e40p_{self.model}/{self.output}_{self.prompt}"
            output_dir += "_crv" if crv_start else ""
            bot = ChatBot(model=self.model, base_url=self.base_url, temperature=self.temperature)
            cv32e40p_runner = RunCV32E40P(
                bot=bot,
                output=output_dir,
                iter=self.iter,
                prompt_type=self.prompt,
                options=self.options,
            )
            cv32e40p_runner.run(crv_start=crv_start)
        else:
            raise ValueError(f"unknown method: {self.method}")
