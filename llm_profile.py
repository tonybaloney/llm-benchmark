from collections import defaultdict
from typing import Any

import click
import llm
from pydantic import BaseModel
from rich.progress import track

try:
    from statistics import fmean
except ImportError: # Python 3.6-3.7 doesn't have fmean
    from statistics import mean as fmean  # YOLO
from time import perf_counter

from rich.box import HEAVY_HEAD, MARKDOWN
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Some of this code is copied from rich-bench, which is my own tool anyway, so 🤷🤷


class BenchmarkData(BaseModel):
    model_name: str
    time_to_first_chunk: float
    total_time: float
    length_of_response: int
    n_chunks: int = 1
    chunks_per_sec: float = 0.0

def build_options(cli_options, model) -> dict[str, Any]:
    validated_options = {}
    if cli_options:
        validated_options = {
            key: value
            for key, value in model.Options(**dict(cli_options))
            if value is not None
        }

    return validated_options

@llm.hookimpl
def register_commands(cli):
    @cli.command("benchmark", short_help="Benchmark one or many models")
    @click.argument("prompt", nargs=1, type=str)
    @click.option("--model", "-m", "models", multiple=True, help="Model names to benchmark")
    @click.option("--no-stream", is_flag=True, help="Do not stream output (no benchmark for TTFT)")
    @click.option("-s", "--system", help="System prompt to use")
    @click.option("--repeat", default=1, help="Number of times to repeat the benchmark")
    @click.option("--markdown", help="Use markdown table format", is_flag=True)
    @click.option(
        "options",
        "-o",
        "--option",
        type=(str, str),
        multiple=True,
        help="key/value options for all models",
    )
    def benchmark(prompt: str, models: tuple[str, ...], no_stream: bool, system: str, repeat: int, markdown: bool, options: tuple[tuple[str, str], ...]):
        # Resolve the models

        resolved_models = {model: llm.get_model(model) for model in models}
        stats = defaultdict(list)

        if repeat < 1:
            repeat = 1  # nice try

        console = Console()

        for _ in track(range(repeat), description="Running Benchmarks...", console=console):

            for model_name, model in resolved_models.items():
                kwargs = build_options(options, model)

                should_stream = model.can_stream and not no_stream
                if not should_stream:
                    kwargs["stream"] = False
                start = perf_counter()
                response = model.prompt(
                    prompt,
                    system=system,
                    # Things we don't support here
                    # fragments=resolved_fragments,
                    # attachments=resolved_attachments,
                    # schema=schema,
                    # system_fragments=resolved_system_fragments,
                    **kwargs,
                )
                n_chunks = 0
                first_chunk_start = None
                if should_stream:
                    first_chunk = True
                    text = ""
                    for chunk in response:
                        if first_chunk:
                            first_chunk_start = perf_counter()
                            first_chunk = False
                        text += chunk
                        n_chunks += 1
                        ... # TODO Collect time per chunk
                else:
                    text = response.text()
                    n_chunks = 1

                end = perf_counter()
                if first_chunk_start is None:   # Errored
                    first_chunk_start = end
                total_time = end - start

                stats[model_name].append(
                    BenchmarkData(
                        model_name=model_name,
                        time_to_first_chunk=first_chunk_start - start if should_stream else 0,
                        total_time=total_time,
                        length_of_response=len(text),
                        n_chunks=n_chunks,
                        chunks_per_sec=n_chunks / total_time if total_time > 0 else 0,
                    )
                )
        create_benchmark_table(console, stats, repeat, models, no_stream, markdown)

def get_col_color(model_name, best_model_name, worst_model_name):
    if model_name == best_model_name:
        return "green"
    elif model_name == worst_model_name:
        return "red"
    return None

def min_max_mean(values):
    return (min(values), max(values), fmean(values))

def create_benchmark_table(console, stats, repeat, models, no_stream, markdown):
    box = MARKDOWN if markdown else HEAVY_HEAD
    table = Table(title=f"Benchmarks, repeat={repeat}, number={len(models)}", box=box)

    col_width = 25

    table.add_column("Benchmark", justify="right", style="cyan", no_wrap=True)
    table.add_column("Total Time", min_width=col_width)
    table.add_column("Time to First Chunk", min_width=col_width)
    table.add_column("Length of Response", min_width=col_width)
    table.add_column("Number of Chunks", min_width=col_width)
    table.add_column("Chunks per Second", min_width=col_width)

    # Repeated benchmarks and single-pass is quite different, don't so min/max/mean for single pass. Keep the UI simple

    # Calculate model with worst and best stats so we can color them red/green
    best_model_total_time = min(stats.items(), key=lambda x: fmean(stat.total_time for stat in x[1]))[0]
    worst_model_total_time = max(stats.items(), key=lambda x: fmean(stat.total_time for stat in x[1]))[0]

    best_model_ttfc = min(stats.items(), key=lambda x: fmean(stat.time_to_first_chunk for stat in x[1]))[0]
    worst_model_ttfc = max(stats.items(), key=lambda x: fmean(stat.time_to_first_chunk for stat in x[1]))[0]

    best_model_cps = max(stats.items(), key=lambda x: fmean(stat.chunks_per_sec for stat in x[1]))[0]
    worst_model_cps = min(stats.items(), key=lambda x: fmean(stat.chunks_per_sec for stat in x[1]))[0]

    if repeat == 1:
        for model_name, model_stats in stats.items():
            table.add_row(
                model_name,
                Text(f"{model_stats[0].total_time:.2f}", style=get_col_color(model_name, best_model_total_time, worst_model_total_time)),
                Text(f"{model_stats[0].time_to_first_chunk:.2f}" if not no_stream else "-", style=get_col_color(model_name, best_model_ttfc, worst_model_ttfc)),
                f"{model_stats[0].length_of_response}",
                f"{model_stats[0].n_chunks}",
                Text(f"{model_stats[0].chunks_per_sec:.2f}" if model_stats[0].chunks_per_sec > 0 else "-", style=get_col_color(model_name, best_model_cps, worst_model_cps))
            )
    else:
        for model_name, model_stats in stats.items():
            total_time_min, total_time_max, total_time_mean = min_max_mean([stat.total_time for stat in model_stats])
            if not no_stream:
                ttfc_min, ttfc_max, ttfc_mean = min_max_mean([stat.time_to_first_chunk for stat in model_stats])
            len_min, len_max, len_mean = min_max_mean([stat.length_of_response for stat in model_stats])
            n_chunks_min, n_chunks_max, n_chunks_mean = min_max_mean([stat.n_chunks for stat in model_stats])
            cps_min, cps_max, cps_mean = min_max_mean([stat.chunks_per_sec for stat in model_stats])

            table.add_row(
                model_name,
                Text(f"{total_time_min:.2f} <-> {total_time_max:.2f} (x̄={total_time_mean:.2f})", style=get_col_color(model_name, best_model_total_time, worst_model_total_time)),
                Text(f"{ttfc_min:.2f} <-> {ttfc_max:.2f} (x̄={ttfc_mean:.2f})" if not no_stream else "-", 
                     style=get_col_color(model_name, best_model_ttfc, worst_model_ttfc)),
                f"{len_min} <-> {len_max} (x̄={len_mean:.2f})",
                f"{n_chunks_min} <-> {n_chunks_max} (x̄={n_chunks_mean:.2f})",
                Text(f"{cps_min:.2f} <-> {cps_max:.2f} (x̄={cps_mean:.2f})" if cps_mean > 0 else "-", style=get_col_color(model_name, best_model_cps, worst_model_cps))
            )
        console.print("Key: Best Model (green), Worst Model (red). (min <-> max (x̄=mean)")

    console.print(table)
