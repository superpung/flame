# FLAME: Enhancing Functional Coverage in Processor Verification using Large Language Models

## Setup

1. Create `.env` file in the root directory
2. Put processor design under test in `dut` directory
3. Install the toolchain of the processor design under test

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m flame -h
```

## Example

```bash
python -m flame --method llm --prompt cot --dataset ibex_v1 --model phind-34b --iter 100 --output example
```
