# pip install numpy
# pip install pandas
# pip install playwright
# playwright install chromium
# pip install anthropic
# pip install python-dotenv

from run_task import AgenticBrowser
from call_llm import call_llm
from helpers import parse_str_to_list

import asyncio
from playwright.async_api import async_playwright, Browser
import pandas as pd
import re
from typing import Optional
from dataclasses import dataclass

MAX_STEPS = 50
NUM_WORKERS = 2


@dataclass
class Task:
    goal: str
    objectives: list[str] | None
    row_info: str | None
    starting_url: str
    sample_id: Optional[str] = None


async def worker(queue: asyncio.Queue, browser: Browser, task_name: str, shared_results: dict[str, list[str]], lock: asyncio.Lock) -> None:
    """Worker function to process tasks from the queue using the AgenticBrowser."""
    while True:
        task: Task = await queue.get()
        if task is None:
            queue.task_done()
            break

        context = await browser.new_context()

        try:
            agent = AgenticBrowser(task.sample_id, context, task_name, shared_results, lock, MAX_STEPS)
            await agent.run_task(task.goal, task.objectives, task.row_info, task.starting_url)

        except Exception as e:
            print(e)

        finally:
            await context.close()
            queue.task_done()


async def dispatch_tasks(tasks: list[Task], task_name: str, shared_results: dict[str, list[str]], lock: asyncio.Lock) -> None:
    """Dispatch tasks to worker coroutines using an asyncio queue."""
    queue = asyncio.Queue()

    for i, task in enumerate(tasks):
        task.sample_id = str(i)
        await queue.put(task)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)

        workers = [
            asyncio.create_task(worker(queue, browser, task_name, shared_results, lock))
            for _ in range(NUM_WORKERS)
        ]

        # Wait until all tasks are processed
        await queue.join()

        # Stop workers
        for _ in workers:
            await queue.put(None)

        await asyncio.gather(*workers)
        await browser.close()


def determine_if_output_csv(user_prompt: str) -> bool:
    """Determine if the task requires an output CSV based on the user prompt."""
    prompt = f"Does the following task ask for an output table/CSV? Answer only YES or NO.\n\nTask:\n{user_prompt}"
    response = call_llm(prompt)

    need_output_csv = re.search(r'\byes\b', response, re.IGNORECASE) is not None

    print(f"Need output CSV?: {need_output_csv}")
    return need_output_csv


def determine_output_csv_columns(user_prompt: str) -> bool:
    """Determine the column names for the output CSV based on the user prompt."""
    prompt = f'List the column names needed for an output table for the following task. Format response as a valid python list eg. ["1", "2"]\n\nTask:\n{user_prompt}'
    response = call_llm(prompt)

    col_names = parse_str_to_list(response)

    print(f"Output CSV column names: {col_names}")
    return col_names


def determine_starting_url(user_prompt: str, row_info: str | None = None) -> str:
    """Determine the starting URL for a task, optionally incorporating row-specific information."""
    prompt = f"What is the starting URL for the following task? Return ONLY the URL, no other text.\n\nTask:\n{user_prompt}"

    if row_info is not None:
        prompt += f"\nComplete the task for the following row in the table:\n{row_info}"

    starting_url = call_llm(prompt)
    print(f"Starting URL: {starting_url}")
    return starting_url


def decompose_tasks(user_prompt: str, objectives: list[str] | None, starting_url: str | None, df: pd.DataFrame) -> list[dict[str, str]]:
    """Decompose the main task into subtasks for each row in the dataframe, incorporating row-specific information."""
    tasks = []

    for _, row in df.iterrows():
        # Create a mini dataframe with only header + this row
        mini_df = pd.DataFrame([row], columns=df.columns)
        row_info = mini_df.to_markdown(index=False)

        if starting_url is None:
            task_starting_url = determine_starting_url(user_prompt, row_info)
        else:
            task_starting_url = starting_url

        tasks.append(
            Task(
                goal=user_prompt,
                objectives=objectives.copy() if objectives is not None else None,
                row_info=row_info,
                starting_url=task_starting_url,
            ),
        )

    return tasks


def generate_plan(user_prompt: str, need_output_csv: bool, first_row: str | None = None) -> str | None:
    """If there are multiple objectives, return a list of objectives. If there is only one objective, return None to just use the user prompt as the objective."""
    input_csv_instruction = f" for a given row in a provided CSV file. For example:\n{first_row}\n\n" if first_row else ". "
    output_csv_instruction = "Results will automatically be written to a CSV table." if need_output_csv else ""

    prompt = f"""List the individual itemized objectives in the "Task" below{input_csv_instruction}Format response as a valid python list eg. ["1", "2"]. There might be only one objective.

Task:
{user_prompt}
{output_csv_instruction}"""
    
    objectives_str = call_llm(prompt)
    objectives = parse_str_to_list(objectives_str) or []
    print(f"Objectives: {objectives}")

    # Only one or two objectives so just use user prompt
    if len(objectives) < 3:
        return None

    return objectives


async def main(user_prompt: str, task_name: str, starting_url: str | None, input_csv_path: str | None = None) -> None:
    """Main function to set up tasks and dispatch them to workers."""
    shared_results = {}
    lock = asyncio.Lock()

    need_output_csv = determine_if_output_csv(user_prompt)
    if need_output_csv:
        output_csv_colnames = determine_output_csv_columns(user_prompt)
        shared_results["header"] = output_csv_colnames

    if input_csv_path is None:
        if starting_url is None:
            starting_url = determine_starting_url(user_prompt)

        objectives = generate_plan(user_prompt, need_output_csv)
        tasks = [
            Task(
                goal=user_prompt,
                objectives=objectives,
                row_info=None,
                starting_url=starting_url,
            ),
        ]
    else:
        df = pd.read_csv(input_csv_path)
        first_row = df.head(1).to_markdown()
        objectives = generate_plan(user_prompt, need_output_csv, first_row)
        tasks = decompose_tasks(user_prompt, objectives, starting_url, df)

    await dispatch_tasks(tasks, task_name, shared_results, lock)


if __name__ == "__main__":
    print("Starting...")
    task_name = 'three_ski_resorts'

    starting_url = None
    csv_path = "C:\\Users\\tomas\\Documents\\andera_browser_agent_assignment\\ski_resorts.csv"
    user_prompt = "Find how many lifts are at each resort. How many km of runs does it have? how many km are green? Put the results in a table"

    asyncio.run(
        main(user_prompt, task_name, starting_url, csv_path)
    )
