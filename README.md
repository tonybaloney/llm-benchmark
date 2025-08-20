# LLM Benchmarking Plugin

This is a plugin for the [llm](https://llm.datasette.io) tool that adds a `benchmark` command to compare the performance of different language models.

The commands runs a prompt with optional system prompt for several models and compares the performance between models.

## Installation

You can install the plugin using pip:

```bash
pip install llm-profile
```

or using `llm`

```bash
llm install llm-profile
```

## Benchmark Usage

To run a benchmark, provide the prompt along with any number of models using the llm alias (from `llm models`):

```bash
$ llm benchmark -m azure/ant-grok-3-mini -m azure/ants-gpt-4.1-mini -m github/gpt-4.1-mini -m github/gpt-4.1-nano -s "Respond in emoji" "Give me a friendly hello message" --markdown
```

For a single pass (no repeats) you will get a summary table:


|               Benchmark | Total Time      | Time to First Chunk | Length of Response | Number of Chunks |
|-------------------------|-----------------|---------------------|--------------------|------------------|
|   azure/ant-grok-3-mini | 6.96            | 6.96                | 3                  | 3                |
| azure/ants-gpt-4.1-mini | 2.53            | 2.37                | 53                 | 15               |
|     github/gpt-4.1-mini | 2.29            | 2.29                | 53                 | 15               |
|     github/gpt-4.1-nano | 2.24            | 2.24                | 52                 | 17               |

To repeat each benchmark and get an average of times, use the `--repeat` argument:

|               Benchmark | Total Time                | Time to First Chunk       | Length of Response        | Number of Chunks          |
|-------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
|   azure/ant-grok-3-mini | 3.71 <-> 8.82 (x̄=6.26)    | 3.69 <-> 8.81 (x̄=6.25)    | 2 <-> 2 (x̄=2.00)          | 2 <-> 2 (x̄=2.00)          |
| azure/ants-gpt-4.1-mini | 0.52 <-> 3.31 (x̄=1.91)    | 0.34 <-> 3.13 (x̄=1.73)    | 53 <-> 54 (x̄=53.50)       | 15 <-> 15 (x̄=15.00)       |
|     github/gpt-4.1-mini | 1.79 <-> 2.33 (x̄=2.06)    | 1.78 <-> 2.33 (x̄=2.06)    | 53 <-> 53 (x̄=53.00)       | 15 <-> 16 (x̄=15.50)       |
|     github/gpt-4.1-nano | 1.82 <-> 2.06 (x̄=1.94)    | 1.82 <-> 2.06 (x̄=1.94)    | 3 <-> 3 (x̄=3.00)          | 4 <-> 4 (x̄=4.00)          |

The printout is a range (min <-> max (x̄=mean))

### Markdown formatted results

By default, tables are printed with color showing the fastest and slowest metric in a benchmark:

![benchmark screenshot](docs/res/screenshot.png)

If you want to customize the output, you can use the `--markdown` flag to get the results in a Markdown-friendly format.

### Non-Streaming models

If you want to benchmark models that do not support streaming, you can use the `--no-stream` flag. This will disable streaming and provide a single response time.

