import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import click
import llm
import yaml
from llm.models import EmbeddingModel
from pydantic import BaseModel
from rich.box import HEAVY_HEAD, MARKDOWN
from rich.progress import track

try:
    from statistics import fmean
except ImportError:  # Python 3.6-3.7 doesn't have fmean
    from statistics import mean as fmean  # YOLO
from time import perf_counter

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
    text: str = ""


class TestModel(BaseModel):
    name: str
    model: str
    system: Optional[str] = None
    prompt: Optional[str] = None  # Inherits from TestCase otherwise
    options: Optional[Dict[str, Any]] = None

    def __hash__(self) -> int:
        return hash(self.name)


class LoadableMixin:
    @classmethod
    def load_file(cls, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        return cls.model_validate(data)  # pyright: ignore[reportAttributeAccessIssue]


class TestPlan(BaseModel, LoadableMixin):
    name: str
    models: List[TestModel]
    repeat: int = 1
    # These apply to all models unless overridden
    system: Optional[str] = None
    prompt: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class BenchmarkEmbedData(BaseModel):
    model_name: str
    total_time: float


class TestEmbedModel(BaseModel):
    name: str
    model: str

    def __hash__(self) -> int:
        return hash(self.name)


class TestEmbedPlan(BaseModel, LoadableMixin):
    name: str
    models: List[TestEmbedModel]
    repeat: int = 1
    data: str


def build_plan_from_args(
    prompt: str, system: Optional[str], models: Iterable[llm.Model], options: tuple[tuple[str, str], ...], repeat: int
) -> TestPlan:
    return TestPlan(
        name="CLI Test Plan",
        models=[
            TestModel(
                name=model.model_id,
                model=model.model_id,
                options=dict(options),
            )
            for model in models
        ],
        system=system,
        prompt=prompt,
        repeat=repeat,
    )


def build_embed_plan_from_args(data: str, models: Iterable[EmbeddingModel], repeat: int) -> TestEmbedPlan:
    return TestEmbedPlan(
        name="CLI Embed Plan",
        models=[
            TestEmbedModel(
                name=model.model_id,
                model=model.model_id,
            )
            for model in models
        ],
        repeat=repeat,
        data=data,
    )


def build_options(cli_options, model) -> dict[str, Any]:
    validated_options = {}
    if cli_options:
        validated_options = {key: value for key, value in model.Options(**dict(cli_options)) if value is not None}

    return validated_options


StatsData = dict[TestModel, list[BenchmarkData]]
EmbedStatsData = dict[TestEmbedModel, list[BenchmarkEmbedData]]


@llm.hookimpl
def register_commands(cli):
    @cli.command("benchmark", short_help="Benchmark one or many models")
    @click.argument("prompt", type=str, required=False)
    @click.option("--model", "-m", "models", multiple=True, help="Model names to benchmark")
    @click.option("--no-stream", is_flag=True, help="Do not stream output (no benchmark for TTFT)")
    @click.option("-s", "--system", help="System prompt to use")
    @click.option("--repeat", default=1, help="Number of times to repeat the benchmark")
    @click.option("--markdown", help="Use markdown table format", is_flag=True)
    @click.option("-g", "--graph", help="Save a graph of the results to the file path provided", type=str)
    @click.option("-p", "--plan", help="Path to a test plan YAML file", type=str, required=False)
    @click.option("--output", help="Path to a YAML file containing the results", type=str, required=False)
    @click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
    @click.option("--no-repeat-first", is_flag=True, help="Do not repeat the first test")
    @click.option(
        "options",
        "-o",
        "--option",
        type=(str, str),
        multiple=True,
        help="key/value options for all models",
    )
    def benchmark(
        prompt: str,
        models: tuple[str, ...],
        no_stream: bool,
        system: str,
        repeat: int,
        markdown: bool,
        options: tuple[tuple[str, str], ...],
        graph: str,
        plan: str,
        output: str,
        verbose: bool,
        no_repeat_first: bool,
    ):
        # Resolve the models

        if prompt:
            if repeat < 1:
                repeat = 1  # nice try
            test_plan = build_plan_from_args(
                prompt, system, [llm.get_model(model) for model in models], options, repeat
            )
        else:
            if not plan:
                raise ValueError("No prompt or --plan provided")
            test_plan = TestPlan.load_file(plan)

        console = Console()

        if verbose:
            console.print("Plan:")
            console.print(test_plan)

        stats = execute_plan(test_plan, console, no_stream=no_stream, verbose=verbose, repeat_first=not no_repeat_first)

        create_benchmark_table(console, stats, test_plan.repeat, no_stream, markdown)

        if graph:
            plot_stats_boxplots(stats, save_path=graph)
            if verbose:
                console.print(f"Saved plot as {graph}")

        if output:
            result_stats = {k.name: [r.model_dump() for r in v] for k, v in stats.items()}
            with open(output, "w") as f:
                yaml.dump(result_stats, f)
            if verbose:
                console.print(f"Saved results as {output}")

    @cli.command("embed-benchmark", short_help="Benchmark one or many embedding models")
    @click.argument("data", type=str, required=False)
    @click.option("--model", "-m", "models", multiple=True, help="Model names to benchmark")
    @click.option("--repeat", default=1, help="Number of times to repeat the benchmark")
    @click.option("--markdown", help="Use markdown table format", is_flag=True)
    @click.option("-g", "--graph", help="Save a graph of the results to the file path provided", type=str)
    @click.option("-p", "--plan", help="Path to a test plan YAML file", type=str, required=False)
    @click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
    @click.option("--no-repeat-first", is_flag=True, help="Do not repeat the first test")
    def embed_benchmark(
        data: str,
        models: tuple[str, ...],
        repeat: int,
        markdown: bool,
        graph: str,
        plan: str,
        verbose: bool,
        no_repeat_first: bool,
    ):
        # Resolve the models

        if data:
            if repeat < 1:
                repeat = 1  # nice try
            test_plan = build_embed_plan_from_args(data, [llm.get_embedding_model(model) for model in models], repeat)
        else:
            if not plan:
                raise ValueError("No input or --plan provided")
            test_plan = TestEmbedPlan.load_file(plan)

        console = Console()

        if verbose:
            console.print("Plan:")
            console.print(test_plan)

        stats = execute_embed_plan(test_plan, console, verbose=verbose, repeat_first=not no_repeat_first)

        create_embed_benchmark_table(console, stats, test_plan.repeat, markdown)

        if graph:
            plot_embed_boxplot(stats, save_path=graph)
            if verbose:
                console.print(f"Saved plot as {graph}")


def execute_test(
    test_model: TestModel, plan: TestPlan, console: Console, no_stream: bool, verbose: bool
) -> BenchmarkData:
    model = llm.get_model(test_model.model)

    kwargs = build_options(plan.options.copy() if plan.options else {}, model)
    kwargs.update(build_options(test_model.options if test_model.options else {}, model))

    should_stream = model.can_stream and not no_stream
    if not should_stream:
        kwargs["stream"] = False
    if verbose:
        console.print(f"Prompting model {test_model.name} with options: {kwargs}")
    start = perf_counter()
    response = model.prompt(
        test_model.prompt if test_model.prompt else plan.prompt,
        system=test_model.system if test_model.system else plan.system,
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
            logging.error(f"Error occurred while streaming response from model {test_model.name}: {e}")

    else:
        try:
            text = response.text()
            n_chunks = 1
        except Exception as e:
            logging.error(f"Error occurred while getting response text from model {test_model.name}: {e}")

    end = perf_counter()
    if first_chunk_start is None:  # Errored
        first_chunk_start = end
    total_time = end - start
    return BenchmarkData(
        model_name=test_model.name,
        time_to_first_chunk=first_chunk_start - start if should_stream else 0,
        total_time=total_time,
        length_of_response=len(text),
        n_chunks=n_chunks,
        chunks_per_sec=n_chunks / total_time if total_time > 0 else 0,
        text=text,
    )


def execute_embed_test(
    test_model: TestEmbedModel, plan: TestEmbedPlan, console: Console, verbose: bool
) -> BenchmarkEmbedData:
    model = llm.get_embedding_model(test_model.model)

    if verbose:
        console.print(f"Embedding model {test_model.name}")
    start = perf_counter()
    try:
        _ = model.embed(plan.data)
    except Exception as e:
        logging.error(f"Error occurred while getting response text from model {test_model.name}: {e}")

    end = perf_counter()
    total_time = end - start
    return BenchmarkEmbedData(
        model_name=test_model.name,
        total_time=total_time,
    )


def execute_plan(
    plan: TestPlan, console: Console, no_stream: bool = False, verbose: bool = False, repeat_first=True
) -> StatsData:
    stats: StatsData = defaultdict(list)
    first = True
    for _ in track(range(plan.repeat), description="Running Benchmarks...", console=console):
        for test_model in plan.models:
            if first and repeat_first:
                execute_test(test_model, plan, console, no_stream, verbose)
                first = False
            stats[test_model].append(execute_test(test_model, plan, console, no_stream, verbose))
    return stats


def execute_embed_plan(
    plan: TestEmbedPlan, console: Console, verbose: bool = False, repeat_first=True
) -> EmbedStatsData:
    stats: EmbedStatsData = defaultdict(list)
    first = True
    for _ in track(range(plan.repeat), description="Running Benchmarks...", console=console):
        for test_model in plan.models:
            if first and repeat_first:
                execute_embed_test(test_model, plan, console, verbose)
                first = False
            stats[test_model].append(execute_embed_test(test_model, plan, console, verbose))
    return stats


def get_col_color(model_name: str, best_model_name: str, worst_model_name: str) -> Optional[str]:
    if model_name == best_model_name:
        return "green"
    elif model_name == worst_model_name:
        return "red"
    return None


def min_max_mean(values: Iterable[float]) -> tuple[float, float, float]:
    return (min(values), max(values), fmean(values))


def create_benchmark_table(console: Console, stats: StatsData, repeat: int, no_stream: bool, markdown: bool):
    box = MARKDOWN if markdown else HEAVY_HEAD
    table = Table(title=f"Benchmarks, repeat={repeat}, number={len(stats)}", box=box)

    table.add_column("Benchmark", justify="right", style="cyan", no_wrap=True)
    table.add_column("Total Time", min_width=25)
    table.add_column("Time to First Chunk", min_width=25)
    table.add_column("Length of Response", min_width=20)
    table.add_column("Number of Chunks", min_width=20)
    table.add_column("Chunks per Second", min_width=20)

    # Repeated benchmarks and single-pass is quite different, don't so min/max/mean for single pass. Keep the UI simple

    # Calculate model with worst and best stats so we can color them red/green
    best_model_total_time = min(stats.items(), key=lambda x: fmean(stat.total_time for stat in x[1]))[0]
    worst_model_total_time = max(stats.items(), key=lambda x: fmean(stat.total_time for stat in x[1]))[0]

    best_model_ttfc = min(stats.items(), key=lambda x: fmean(stat.time_to_first_chunk for stat in x[1]))[0]
    worst_model_ttfc = max(stats.items(), key=lambda x: fmean(stat.time_to_first_chunk for stat in x[1]))[0]

    best_model_cps = max(stats.items(), key=lambda x: fmean(stat.chunks_per_sec for stat in x[1]))[0]
    worst_model_cps = min(stats.items(), key=lambda x: fmean(stat.chunks_per_sec for stat in x[1]))[0]

    if repeat == 1:
        for test_model, model_stats in stats.items():
            table.add_row(
                test_model.name,
                Text(
                    f"{model_stats[0].total_time:.2f}",
                    style=(
                        get_col_color(test_model.name, best_model_total_time.name, worst_model_total_time.name) or ""
                    ),
                ),
                Text(
                    f"{model_stats[0].time_to_first_chunk:.2f}" if not no_stream else "-",
                    style=(get_col_color(test_model.name, best_model_ttfc.name, worst_model_ttfc.name) or ""),
                ),
                f"{model_stats[0].length_of_response}",
                f"{model_stats[0].n_chunks}",
                Text(
                    f"{model_stats[0].chunks_per_sec:.2f}" if model_stats[0].chunks_per_sec > 0 else "-",
                    style=(get_col_color(test_model.name, best_model_cps.name, worst_model_cps.name) or ""),
                ),
            )
    else:
        for test_model, model_stats in stats.items():
            total_time_min, total_time_max, total_time_mean = min_max_mean([stat.total_time for stat in model_stats])
            ttfc_min = ttfc_max = ttfc_mean = 0
            if not no_stream:
                ttfc_min, ttfc_max, ttfc_mean = min_max_mean([stat.time_to_first_chunk for stat in model_stats])
            len_min, len_max, len_mean = min_max_mean([stat.length_of_response for stat in model_stats])
            n_chunks_min, n_chunks_max, n_chunks_mean = min_max_mean([stat.n_chunks for stat in model_stats])
            cps_min, cps_max, cps_mean = min_max_mean([stat.chunks_per_sec for stat in model_stats])

            table.add_row(
                test_model.name,
                Text(
                    f"{total_time_min:.2f} <-> {total_time_max:.2f} (x̄={total_time_mean:.2f})",
                    style=(
                        get_col_color(test_model.name, best_model_total_time.name, worst_model_total_time.name) or ""
                    ),
                ),
                Text(
                    f"{ttfc_min:.2f} <-> {ttfc_max:.2f} (x̄={ttfc_mean:.2f})" if not no_stream else "-",
                    style=(get_col_color(test_model.name, best_model_ttfc.name, worst_model_ttfc.name) or ""),
                ),
                f"{len_min} <-> {len_max} (x̄={len_mean:.1f})",
                f"{n_chunks_min} <-> {n_chunks_max} (x̄={n_chunks_mean:.1f})",
                Text(
                    f"{cps_min:.2f} <-> {cps_max:.2f} (x̄={cps_mean:.2f})" if cps_mean > 0 else "-",
                    style=(get_col_color(test_model.name, best_model_cps.name, worst_model_cps.name) or ""),
                ),
            )
        console.print("Key: Best Model (green), Worst Model (red). (min <-> max (x̄=mean)")

    console.print(table)


def create_embed_benchmark_table(console: Console, stats: EmbedStatsData, repeat: int, markdown: bool):
    box = MARKDOWN if markdown else HEAVY_HEAD
    table = Table(title=f"Benchmarks, repeat={repeat}, number={len(stats)}", box=box)

    table.add_column("Benchmark", justify="right", style="cyan", no_wrap=True)
    table.add_column("Total Time", min_width=25)

    # Repeated benchmarks and single-pass is quite different, don't so min/max/mean for single pass. Keep the UI simple

    # Calculate model with worst and best stats so we can color them red/green
    best_model_total_time = min(stats.items(), key=lambda x: fmean(stat.total_time for stat in x[1]))[0]
    worst_model_total_time = max(stats.items(), key=lambda x: fmean(stat.total_time for stat in x[1]))[0]

    if repeat == 1:
        for test_model, model_stats in stats.items():
            table.add_row(
                test_model.name,
                Text(
                    f"{model_stats[0].total_time:.2f}",
                    style=(
                        get_col_color(test_model.name, best_model_total_time.name, worst_model_total_time.name) or ""
                    ),
                ),
            )
    else:
        for test_model, model_stats in stats.items():
            total_time_min, total_time_max, total_time_mean = min_max_mean([stat.total_time for stat in model_stats])

            table.add_row(
                test_model.name,
                Text(
                    f"{total_time_min:.2f} <-> {total_time_max:.2f} (x̄={total_time_mean:.2f})",
                    style=(
                        get_col_color(test_model.name, best_model_total_time.name, worst_model_total_time.name) or ""
                    ),
                ),
            )
        console.print("Key: Best Model (green), Worst Model (red). (min <-> max (x̄=mean)")

    console.print(table)


def _prepare_datasets(stats: StatsData) -> tuple[list[tuple[str, str]], list[list[float]], list[str], list[str]]:
    """Return model_names, datasets, titles and ylabels prepared from stats."""
    models = list(stats.keys())
    if not models:
        raise ValueError("stats must contain at least one model with benchmark data")

    total_time_data = [[d.total_time for d in stats[m]] for m in models]
    ttfc_data = [[d.time_to_first_chunk for d in stats[m]] for m in models]
    length_data = [[d.length_of_response for d in stats[m]] for m in models]
    cps_data = [[d.chunks_per_sec for d in stats[m]] for m in models]

    datasets = [total_time_data, ttfc_data, length_data, cps_data]
    titles = ["Total Time (s)", "Time to First Chunk (s)", "Length of Response", "Chunks per Second"]
    ylabels = ["Seconds", "Seconds", "Characters", "Chunks/sec"]

    # To keep the x-labels small, use a key (A, B, C, D etc.) then use that key in the legend
    return (
        [
            (chr(65 + key_offset), f"{chr(65 + key_offset)}: {model.name[:40]}")
            for key_offset, model in enumerate(models)
        ],
        datasets,
        titles,
        ylabels,
    )


ColorMappedColor = tuple[float, float, float, float]  # RGBA tuple


def _build_colors(model_names: list[tuple[str, str]]) -> tuple[list[ColorMappedColor], dict[str, ColorMappedColor]]:
    """Return a list of colors and a name->color mapping for the models."""

    from matplotlib import cm

    cmap = cm.get_cmap("tab10" if len(model_names) <= 10 else "tab20")

    if len(model_names) == 1:
        colors = [cmap(0.0)]  # black
    else:
        max_idx = max(1, len(model_names) - 1)
        colors = [cmap(i / max_idx) for i in range(len(model_names))]
    model_to_color = {name[0]: colors[i] for i, name in enumerate(model_names)}
    return colors, model_to_color


def _draw_boxplot(ax, data: list[float], labels: list[str], colors: list[ColorMappedColor], title: str, ylabel: str):
    """Draw and style a single boxplot on the given Axes."""
    bplot = ax.boxplot(data, patch_artist=True, labels=labels)

    for patch, color in zip(bplot.get("boxes", []), colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.8)

    for whisker in bplot.get("whiskers", []):
        whisker.set_color("black")
        whisker.set_linewidth(1)
    for cap in bplot.get("caps", []):
        cap.set_color("black")
        cap.set_linewidth(1)
    for median in bplot.get("medians", []):
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=45)

    return bplot


def plot_stats_boxplots(stats: StatsData, save_path: str, figsize=(12, 10)):
    """
    Orchestrate creation of the 2x2 boxplots for the given StatsData using the helper functions.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create plots (pip install matplotlib)") from exc

    model_names, datasets, titles, ylabels = _prepare_datasets(stats)

    colors, model_to_color = _build_colors(model_names)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, data, title, ylabel in zip(axes, datasets, titles, ylabels):
        _draw_boxplot(ax, data, [n[0] for n in model_names], colors, title, ylabel)

    legend_handles = [
        mpatches.Patch(facecolor=model_to_color[name[0]], edgecolor="black", label=name[1]) for name in model_names
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(model_names), 6))
    fig.savefig(save_path)


def plot_embed_boxplot(stats: EmbedStatsData, save_path: str, figsize=(10, 8)):
    """
    Create a single boxplot (total_time) for embedding benchmark data.

    - stats: mapping TestEmbedModel -> list[BenchmarkEmbedData]
    - save_path: path to write the PNG
    - figsize: matplotlib figure size
    - show: whether to call plt.show()

    The function uses the same short-key mapping (A:, B:, ...) and a legend that maps
    keys back to full model names, and tries to avoid clipping long legend labels.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create plots (pip install matplotlib)") from exc

    models = list(stats.keys())
    if not models:
        raise ValueError("stats must contain at least one model with benchmark data")

    # Build short keys and legend entries
    model_keys = [(chr(65 + idx), f"{chr(65 + idx)}: {m.name[:60]}") for idx, m in enumerate(models)]

    # Dataset is a list per model of total_time values
    dataset = [[d.total_time for d in stats[m]] for m in models]

    # Build colors and mapping using existing helper
    colors, model_to_color = _build_colors(model_keys)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw the boxplot using the short keys as x labels
    _draw_boxplot(ax, dataset, [k[0] for k in model_keys], colors, "Total Time (s)", "Seconds")

    legend_handles = [mpatches.Patch(facecolor=model_to_color[k[0]], edgecolor="black", label=k[1]) for k in model_keys]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
    )

    fig.savefig(save_path)
