import logging
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

StatsData = dict[str, list[BenchmarkData]]

@llm.hookimpl
def register_commands(cli):
    @cli.command("benchmark", short_help="Benchmark one or many models")
    @click.argument("prompt", nargs=1, type=str)
    @click.option("--model", "-m", "models", multiple=True, help="Model names to benchmark")
    @click.option("--no-stream", is_flag=True, help="Do not stream output (no benchmark for TTFT)")
    @click.option("-s", "--system", help="System prompt to use")
    @click.option("--repeat", default=1, help="Number of times to repeat the benchmark")
    @click.option("--markdown", help="Use markdown table format", is_flag=True)
    @click.option("-g", "--graph", help="Save a graph of the results to the file path provided", type=str)
    @click.option(
        "options",
        "-o",
        "--option",
        type=(str, str),
        multiple=True,
        help="key/value options for all models",
    )
    def benchmark(prompt: str, models: tuple[str, ...], no_stream: bool, system: str, repeat: int, markdown: bool, options: tuple[tuple[str, str], ...], graph: str):
        # Resolve the models

        resolved_models = {model: llm.get_model(model) for model in models}
        stats: StatsData = defaultdict(list)

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
                text = ""
                if should_stream:
                    first_chunk = True
                    try:
                        for chunk in response:
                            if first_chunk:
                                first_chunk_start = perf_counter()
                                first_chunk = False
                            text += chunk
                            n_chunks += 1
                    except Exception as e:
                        logging.error(f"Error occurred while streaming response from model {model_name}: {e}")

                else:
                    try:
                        text = response.text()
                        n_chunks = 1
                    except Exception as e:
                        logging.error(f"Error occurred while getting response text from model {model_name}: {e}")

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

        if graph:
            plot_stats_boxplots(stats, save_path=graph)
            console.print(f"Saved plot as {graph}")


def get_col_color(model_name, best_model_name, worst_model_name):
    if model_name == best_model_name:
        return "green"
    elif model_name == worst_model_name:
        return "red"
    return None

def min_max_mean(values):
    return (min(values), max(values), fmean(values))

def create_benchmark_table(console, stats: StatsData, repeat, models, no_stream, markdown):
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
                Text(f"{model_stats[0].total_time:.2f}", style=(get_col_color(model_name, best_model_total_time, worst_model_total_time) or "")),
                Text(f"{model_stats[0].time_to_first_chunk:.2f}" if not no_stream else "-", style=(get_col_color(model_name, best_model_ttfc, worst_model_ttfc) or "")),
                f"{model_stats[0].length_of_response}",
                f"{model_stats[0].n_chunks}",
                Text(f"{model_stats[0].chunks_per_sec:.2f}" if model_stats[0].chunks_per_sec > 0 else "-", style=(get_col_color(model_name, best_model_cps, worst_model_cps) or ""))
            )
    else:
        for model_name, model_stats in stats.items():
            total_time_min, total_time_max, total_time_mean = min_max_mean([stat.total_time for stat in model_stats])
            ttfc_min = ttfc_max = ttfc_mean = 0
            if not no_stream:
                ttfc_min, ttfc_max, ttfc_mean = min_max_mean([stat.time_to_first_chunk for stat in model_stats])
            len_min, len_max, len_mean = min_max_mean([stat.length_of_response for stat in model_stats])
            n_chunks_min, n_chunks_max, n_chunks_mean = min_max_mean([stat.n_chunks for stat in model_stats])
            cps_min, cps_max, cps_mean = min_max_mean([stat.chunks_per_sec for stat in model_stats])

            table.add_row(
                model_name,
                Text(f"{total_time_min:.2f} <-> {total_time_max:.2f} (x̄={total_time_mean:.2f})", style=(get_col_color(model_name, best_model_total_time, worst_model_total_time) or "")),
                Text(f"{ttfc_min:.2f} <-> {ttfc_max:.2f} (x̄={ttfc_mean:.2f})" if not no_stream else "-", style=(get_col_color(model_name, best_model_ttfc, worst_model_ttfc) or "")),
                f"{len_min} <-> {len_max} (x̄={len_mean:.2f})",
                f"{n_chunks_min} <-> {n_chunks_max} (x̄={n_chunks_mean:.2f})",
                Text(f"{cps_min:.2f} <-> {cps_max:.2f} (x̄={cps_mean:.2f})" if cps_mean > 0 else "-", style=(get_col_color(model_name, best_model_cps, worst_model_cps) or ""))
            )
        console.print("Key: Best Model (green), Worst Model (red). (min <-> max (x̄=mean)")

    console.print(table)

def _import_matplotlib():
    """Dynamically import matplotlib modules so matplotlib is optional at runtime."""
    import importlib
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        mpatches = importlib.import_module("matplotlib.patches")
        cm = importlib.import_module("matplotlib.cm")
        return plt, mpatches, cm
    except Exception as exc:
        raise RuntimeError("matplotlib is required to create plots (pip install matplotlib)") from exc


def _prepare_datasets(stats: StatsData):
    """Return model_names, datasets, titles and ylabels prepared from stats."""
    model_names = list(stats.keys())
    if not model_names:
        raise ValueError("stats must contain at least one model with benchmark data")

    total_time_data = [[d.total_time for d in stats[m]] for m in model_names]
    ttfc_data = [[d.time_to_first_chunk for d in stats[m]] for m in model_names]
    length_data = [[d.length_of_response for d in stats[m]] for m in model_names]
    cps_data = [[d.chunks_per_sec for d in stats[m]] for m in model_names]

    datasets = [total_time_data, ttfc_data, length_data, cps_data]
    titles = ["Total Time (s)", "Time to First Chunk (s)", "Length of Response", "Chunks per Second"]
    ylabels = ["Seconds", "Seconds", "Characters", "Chunks/sec"]

    return model_names, datasets, titles, ylabels


def _build_colors(model_names, cm):
    """Return a list of colors and a name->color mapping for the models."""
    cmap = cm.get_cmap("tab10" if len(model_names) <= 10 else "tab20")
    if len(model_names) == 1:
        colors = [cmap(0.0)]
    else:
        max_idx = max(1, len(model_names) - 1)
        colors = [cmap(i / max_idx) for i in range(len(model_names))]
    model_to_color = {name: colors[i] for i, name in enumerate(model_names)}
    return colors, model_to_color


def _draw_boxplot(ax, data, labels, colors, title, ylabel):
    """Draw and style a single boxplot on the given Axes."""
    bplot = ax.boxplot(data, patch_artist=True, labels=labels)

    for patch, color in zip(bplot.get('boxes', []), colors):
        if hasattr(patch, 'set_facecolor'):
            patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.8)

    for whisker in bplot.get('whiskers', []):
        whisker.set_color("black")
        whisker.set_linewidth(1)
    for cap in bplot.get('caps', []):
        cap.set_color("black")
        cap.set_linewidth(1)
    for median in bplot.get('medians', []):
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.tick_params(axis='x', rotation=45)

    return bplot


def plot_stats_boxplots(stats: StatsData, save_path: str, figsize=(12, 10)):
    """
    Orchestrate creation of the 2x2 boxplots for the given StatsData using the helper functions.
    """
    plt, mpatches, cm = _import_matplotlib()

    model_names, datasets, titles, ylabels = _prepare_datasets(stats)

    colors, model_to_color = _build_colors(model_names, cm)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for ax, data, title, ylabel in zip(axes, datasets, titles, ylabels):
        _draw_boxplot(ax, data, model_names, colors, title, ylabel)

    legend_handles = [mpatches.Patch(facecolor=model_to_color[name], edgecolor='black', label=name) for name in model_names]
    fig.legend(handles=legend_handles, loc='upper center', ncol=min(len(model_names), 6), bbox_to_anchor=(0.5, 1.02))

    fig.savefig(save_path)
