"""Evolve a Hermes Agent skill using DSPy + MIPROv2.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import ast
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)
from dspy.adapters.json_adapter import JSONAdapter
from dspy.utils.exceptions import AdapterParseError

console = Console()


# ── Custom adapter with regex fallback for MiniMax ────────────────────────────

class RobustJSONAdapter(JSONAdapter):
    """JSONAdapter subclass that falls back to regex extraction when JSON parsing fails.

    MiniMax-M2.7 sometimes produces malformed [[ ## reasoning ## ]] / [[ ## output ## ]] blocks
    (e.g. bare Python expressions like ``{pr.get('title', 'N/A')}`` or markdown code blocks).
    Instead of crashing on AdapterParseError, this adapter tries to recover the output field
    using regex before giving up.
    """

    def parse(self, signature, completion):
        try:
            return super().parse(signature, completion)
        except AdapterParseError:
            output_field = list(signature.output_fields.keys())
            if not output_field:
                raise

            # Try regex extraction: look for [[ ## <field_name> ## ]] blocks
            # and use the last one found as the output value.
            field_name = output_field[0]  # typically "output"

            # Try [[ ## <field> ## ]] ... [[ ## <field> ## ]] pattern
            pattern = rf'\[\[ ## {re.escape(field_name)} ## \]\]\s*\n?(.*?)(?=\[\[ ##|\Z)'
            match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)

            if match:
                recovered = match.group(1).strip()
                if recovered:
                    console.print(f"[yellow]  [RobustJSONAdapter] Recovered output via regex fallback ({len(recovered)} chars)[/yellow]")
                    return {field_name: recovered}

            # Last resort: treat the entire completion as the output
            console.print(f"[yellow]  [RobustJSONAdapter] No structured parse; using raw completion ({len(completion)} chars)[/yellow]")
            return {field_name: completion.strip()}


# ── LLM response normalization ─────────────────────────────────────────────

def _normalize_llm_text_response(raw: str) -> str:
    """Normalize LLM output: unwrap {'text': ...} wrappers and strip code fences."""
    text = str(raw).strip()

    # Try to unwrap {"text": ...} / {'text': ...} — only at the start of the string
    if text.startswith("{") or text.startswith("'{"):
        text_like_pattern = re.compile(r"^\s*\{\s*['\"]?text['\"]?\s*:\s*")
        if text_like_pattern.match(text):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict) and "text" in parsed:
                    text = str(parsed["text"])
            except (ValueError, SyntaxError):
                text = re.sub(
                    r"^\s*\{\s*['\"]?text['\"]?\s*:\s*['\"]?",
                    "",
                    text,
                )
                text = re.sub(r"['\"]?\s*\}\s*$", "", text)

    # Strip ONLY outer markdown code fences (anchored to string start/end, not per-line)
    text = re.sub(r'^```(?:\w+)?\s*', '', text).strip()
    text = re.sub(r'\s*```$', '', text).strip()

    return text


# ── LLM-based skill body evolution ───────────────────────────────────────────

def evolve_skill_body(
    original_body: str,
    best_instruction: str,
    few_shot_examples: list[dict],
    config: EvolutionConfig,
    num_examples: int = 3,
) -> str:
    """Regenerate the skill body using an LLM, inspired by the best MIPRO instruction.

    Args:
        original_body: The original skill markdown body (without frontmatter).
        best_instruction: The best instruction string produced by MIPRO.
        few_shot_examples: List of {task_input, expected_behavior, agent_output} dicts.
        config: EvolutionConfig with eval_model set.
        num_examples: How many few-shot examples to include.

    Returns:
        The evolved skill body (markdown), keeping the original frontmatter unchanged.
    """
    examples_text = ""
    for i, ex in enumerate(few_shot_examples[:num_examples]):
        examples_text += f"## Example {i+1}\n\n**Task:**\n{ex.get('task_input', '')}\n\n"
        if 'expected_behavior' in ex:
            examples_text += f"**Expected behavior:**\n{ex['expected_behavior']}\n\n"
        if 'agent_output' in ex:
            examples_text += f"**Agent output:**\n{ex['agent_output']}\n\n"
        examples_text += "---\n\n"

    prompt = f"""You are improving a Hermes Agent skill document. The original skill body is followed by
an optimized instruction discovered through automated experimentation, plus real examples of the skill in action.

Your task: Rewrite the skill body to incorporate the improvements from the optimized instruction,
making the improvements concrete and actionable in the skill text itself. Do NOT just copy the instruction
verbatim — weave its key insights into well-structured skill documentation.

Requirements:
- Keep the same level of detail and structure as the original
- Preserve any numbered steps, tables, or special formatting
- Make the improvements feel natural, not bolted on
- Frontmatter stays unchanged (handled separately)

---
## Original Skill Body
{original_body[:4000]}

---
## Optimized Instruction (from experimentation)
{best_instruction[:2000]}

---
## Examples from evaluation dataset
{examples_text}
---
## Your rewritten skill body:

IMPORTANT: Output ONLY the rewritten skill body in plain markdown. Do NOT wrap in quotes, dictionaries, code fences, or any wrapper. Start directly with the heading "# GitHub Code Review" (or whatever heading is appropriate).
"""

    lm = dspy.LM(config.eval_model)
    with dspy.context(lm=lm):
        response = lm(prompt, n=1, temperature=0.7)
        if isinstance(response, list):
            generated = response[0].text if hasattr(response[0], 'text') else str(response[0])
        else:
            generated = str(response)

        generated = _normalize_llm_text_response(generated)

    if not generated or len(generated) < 100:
        console.print("[yellow]  Skill body evolution produced very short output; keeping original body[/yellow]")
        return original_body

    return generated


# ── Main evolve function ─────────────────────────────────────────────────────

def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run MIPROv2 optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["body"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + MIPROv2 optimizer ──────────────────────────────
    # Use robust adapter to handle MiniMax's occasional malformed output blocks
    dspy.configure(adapter=RobustJSONAdapter())

    # Configure the default LM for evaluation
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Suppress the known MIPROv2 warning about unused fields in InstructSelector
    # (program_code, module, program_description, module_description, previous_instructions)
    # — these are passed by MIPROv2 internally but InstructSelector only uses a subset.
    # Use a broad pattern since the field names vary per call.
    warnings.filterwarnings("ignore", category=UserWarning,
                           message=r"Input contains fields not in signature")

    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: MIPROv2 ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run MIPROv2 optimization ─────────────────────────────────────
    console.print(f"\n[bold cyan]Running MIPROv2 optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        # Try GEPA first (more powerful) but it requires init compatibility
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_steps=iterations,
        )
        optimizer_name = "GEPA"
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric,
            auto="light",
        )
        optimizer_name = "MIPROv2"
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved instruction from MIPRO ───────────────────────
    # The optimized module's skill_text contains the instruction that was optimized
    evolved_instruction = optimized_module.skill_text

    # ── 7. Evolve the full skill body using LLM ─────────────────────────
    console.print(f"\n[bold]Evolving skill body content[/bold]")
    few_shot_examples = []
    for ex in dataset.train[:3]:
        few_shot_examples.append({
            "task_input": getattr(ex, "task_input", ""),
            "expected_behavior": getattr(ex, "expected_behavior", ""),
            "agent_output": getattr(ex, "agent_output", ""),
        })

    evolved_body = evolve_skill_body(
        original_body=skill["body"],
        best_instruction=evolved_instruction,
        few_shot_examples=few_shot_examples,
        config=config,
        num_examples=min(3, len(dataset.train)),
    )
    console.print(f"  Evolved body: {len(evolved_body):,} chars (original: {len(skill['body']):,})")

    # Reassemble the full skill (frontmatter + evolved body)
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 8. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 9. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    # Build a fresh module with the evolved body for holdout evaluation
    evolved_module = SkillModule(evolved_body)
    holdout_examples = dataset.to_dspy_examples("holdout")

    holdout_scores = []
    for ex in holdout_examples:
        try:
            with dspy.context(lm=lm):
                baseline_pred = baseline_module(task_input=getattr(ex, "task_input", ""))
                evolved_pred = evolved_module(task_input=getattr(ex, "task_input", ""))
                baseline_score = skill_fitness_metric(ex, baseline_pred)
                evolved_score = skill_fitness_metric(ex, evolved_pred)
                holdout_scores.append({
                    "example": getattr(ex, "task_input", "")[:60],
                    "baseline_score": baseline_score,
                    "evolved_score": evolved_score,
                })
        except Exception as e:
            task_input_preview = getattr(ex, "task_input", "")[:60]
            logging.exception(
                "Error during holdout evaluation for task_input=%r",
                task_input_preview,
            )
            holdout_scores.append({
                "example": task_input_preview,
                "baseline_score": None,
                "evolved_score": None,
                "error": str(e),
            })

    # Filter out failed evaluations
    valid_baseline = [s["baseline_score"] for s in holdout_scores if s["baseline_score"] is not None]
    valid_evolved = [s["evolved_score"] for s in holdout_scores if s["evolved_score"] is not None]
    avg_baseline = sum(valid_baseline) / len(valid_baseline) if valid_baseline else None
    avg_evolved = sum(valid_evolved) / len(valid_evolved) if valid_evolved else None

    console.print(f"\n  Holdout: {len(valid_evolved)}/{len(holdout_examples)} examples evaluated successfully")

    # ── 10. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change = avg_evolved - avg_baseline
    change_color = "green" if change > 0 else "yellow"
    table.add_row(
        "Holdout Score (avg)",
        f"{avg_baseline:.3f}" if valid_baseline else "N/A",
        f"{avg_evolved:.3f}" if valid_evolved else "N/A",
        f"{change:+.3f}" if valid_evolved else "N/A",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")
    table.add_row("Optimizer", "", optimizer_name, "")

    console.print()
    console.print(table)

    # ── 11. Save output ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "optimizer": optimizer_name,
        "baseline_holdout_score": avg_baseline,
        "evolved_holdout_score": avg_evolved,
        "holdout_evaluated": len(valid_evolved),
        "holdout_total": len(holdout_examples),
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if valid_evolved and avg_evolved > 0.35:
        console.print(f"\n[bold green]✓ Evolution successful — avg holdout score: {avg_evolved:.3f}[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Holdout evaluation had limited success ({len(valid_evolved)}/{len(holdout_examples)} examples)[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of MIPROv2 iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for MIPRO reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run):
    """Evolve a Hermes Agent skill using DSPy + MIPROv2 optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()