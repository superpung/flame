import argparse
import random

from flame.run import Runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--model", required=False, type=str)
    parser.add_argument("--prompt", required=False, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--iter", required=False, type=int)
    parser.add_argument("--output", required=False, type=str)
    parser.add_argument("--base_url", required=False, type=str)
    parser.add_argument("--crv_start", required=False, type=str)
    parser.add_argument("--options", required=False, type=str)
    parser.add_argument("--temperature", required=False, type=float)
    args = parser.parse_args()

    random.seed(0)

    runner = Runner(args=args)
    runner.run()
