from call_llm import call_llm, call_llm_tool_call
from helpers import parse_str_to_json, parse_str_to_list

import asyncio
import base64
from pathlib import Path
from datetime import datetime
from typing import Any
import csv
from playwright.async_api import Page, BrowserContext


class AgenticBrowser:
    def __init__(self, sample_id: str, context: BrowserContext, task_name: str, shared_results: dict[str, list[str]], lock: asyncio.Lock, max_steps: int):
        self.sample_id = sample_id
        self.context = context
        self.task_name = task_name
        self.shared_results = shared_results
        self.lock = lock
        self.max_steps = max_steps
        self.max_context_size = 50
        self.history = []
        self.num_screenshots = 0

        self.sample_directory = Path.cwd() / self.task_name / self.sample_id
        self.debug_directory = self.sample_directory / "debug"
        self.debug_directory.mkdir(exist_ok=True, parents=True)
        self.output_csv_path = Path.cwd() / self.task_name / "results.csv"


    async def get_page_state(self, page: Page) -> tuple[str, list[dict[str, Any]]]:
        """Extract the current page state, including a screenshot and a list of visible interactable elements."""
        await page.wait_for_selector("body", timeout=5000)
       
        # Extract elements first to set the data-agent-ids
        elements = await self.extract_page_elements(page)
        elements_str = self.convert_elements_to_str(elements)
        print(f"{self.sample_id} Found {len(elements)} elements.")
       
        # Add labels to elements on page, take a screenshot, then remove the labels
        await self.add_page_labels(page, elements)
        img_b64 = await self.take_and_save_screenshot(self.debug_directory, page, len(self.history))
        await self.remove_page_labels(page)
       
        return img_b64, elements_str


    async def remove_page_labels(self, page: Page) -> None:
        """Remove previously injected agent labels from the page."""
        await page.evaluate("""
        () => {
            document.querySelectorAll('.agent-label').forEach(el => el.remove());
        }
        """)


    async def extract_page_elements(self, page: Page) -> list[dict[str, Any]]:
        """Extract visible/interactable elements and assign each one a data-agent-id."""
        return await page.evaluate("""() => {
            const result = [];
            const seen = new Set();
            const vh = window.innerHeight;
            const vw = window.innerWidth;

            const elements = Array.from(
                document.querySelectorAll('button, input, a, textarea, [role="button"]')
            );

            elements.forEach((el) => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                const isVisible = rect.width > 0 && rect.height > 0 &&
                                style.display !== 'none' &&
                                style.visibility !== 'hidden';

                const isInsideViewport = rect.top >= 0 && rect.left >= 0 &&
                                        rect.bottom <= vh && rect.right <= vw;

                if (!isVisible || !isInsideViewport) return;

                let text = (el.innerText || el.placeholder || el.getAttribute('aria-label') || el.value || "").trim();
                if (el.tagName === 'A' && !text && !el.querySelector('img')) return;

                const coordKey = `${Math.round(rect.left / 5) * 5}-${Math.round(rect.top / 5) * 5}`;
                if (seen.has(coordKey)) return;
                seen.add(coordKey);

                const id = result.length;

                el.setAttribute('data-agent-id', id);

                result.push({
                    id: id,
                    tag: el.tagName,
                    text: text.substring(0, 60).replace(/\s+/g, ' '),
                    x: Math.round(rect.left + rect.width / 2),
                    y: Math.round(rect.top + rect.height / 2)
                });
            });

            return result;
        }""")


    async def add_page_labels(self, page: Page, elements: list[dict[str, Any]]) -> None:
        """Add visual labels to the page for the extracted elements."""
        await page.evaluate("""
        (elements) => {
            elements.forEach((el) => {
                const label = document.createElement('div');
                label.className = 'agent-label';
                label.innerText = el.id;

                Object.assign(label.style, {
                    position: 'fixed',
                    top: `${el.y}px`,
                    left: `${el.x}px`,
                    background: 'red',
                    color: 'white',
                    fontSize: '10px',
                    fontWeight: 'bold',
                    padding: '2px 4px',
                    zIndex: '9999999',
                    pointerEvents: 'none',
                    borderRadius: '3px',
                    border: '1px solid white'
                });

                document.body.appendChild(label);
            });
        }
        """, elements)


    def convert_elements_to_str(self, elements: list[dict[str, Any]]) -> str:
        """Convert elements list of dicts to a table string."""
        s = "ID | Tag | Text\n"

        for el in elements:
            text = str(el['text'])[:30].replace("\n", " ")
            line = f"{el['id']} | {el['tag']} | {text}"
            s += f"{line}\n"

        return s
    

    async def take_and_save_screenshot(self, directory: str, page: Page, step: int) -> bytes:
        """Saves a screenshot to the sample_id folder."""
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{step}_{timestamp}.jpeg"
        img_path = directory / filename

        # Take the screenshot
        screenshot = await page.screenshot(path=str(img_path), type="jpeg", quality=60)
       
        print(f"{self.sample_id} Saved to: {img_path}")
        return base64.b64encode(screenshot).decode("utf-8")


    def convert_history_to_str(self, context_size: int | None = None) -> str:
        """Convert the history of steps taken to a string format for LLM input."""
        if not self.history:
            return ""
        
        context_size = context_size or self.max_context_size

        steps_taken = "Steps Taken:"
        for i, h in enumerate(self.history[-context_size:]):
            steps_taken += f"\n\nStep {i}) {h.get('reasoning')}\n"

            for k, v in h.items():
                if k in ["reasoning", "url"] or v is None:
                    continue
                steps_taken += f"{k}: {v}, "

        return steps_taken


    def save_history_to_txt(self):
        """Save the history of steps taken to a text file for debugging."""
        steps_taken = self.convert_history_to_str()
        history_filepath = self.debug_directory / "steps_taken.txt"

        with open(history_filepath, "w", encoding="utf-8") as f:
            f.write(steps_taken + "\n")


    async def safe_click(self, page: Page, selector: str):
        """Click an element."""
        await page.click(selector, force=True)
        await page.wait_for_load_state("domcontentloaded")
        print(f"{self.sample_id} Clicked {selector}")


    async def safe_type(self, page: Page, selector: str, value: str):
        """Type into an element and press Enter."""
        await page.fill(selector, value)
        await page.keyboard.press("Enter")
        await page.wait_for_load_state("networkidle")

        print(f"{self.sample_id} Typed {value}")


    async def safe_scroll(self, page: Page):
        """Scroll down the page."""
        before = await page.evaluate("() => window.scrollY")

        await page.mouse.wheel(0, 600)
        await asyncio.sleep(1)

        after = await page.evaluate("() => window.scrollY")

        if after <= before:
            raise Exception("Scroll verification failed")

        print(f"{self.sample_id} Scroll + verified")


    def show_shared_results_row_as_json(self) -> str:
        """Convert the current shared_results row to a JSON string for LLM input, with column names as keys."""
        output_json = {}
        header = self.shared_results["header"]
        row = self.shared_results[self.sample_id]

        for colname, cell in zip(header, row):
            if cell is None:
                output_json[colname] = cell

        return output_json


    async def write_to_shared_results(self) -> None:
        """Prompt the LLM to fill in missing values in the shared_results JSON based on the last step taken, then write the updated values back to shared_results."""
        last_step = self.convert_history_to_str(1)
        row_json = self.show_shared_results_row_as_json()

        prompt = f"""Based on the last Step Taken below, what missing info can to filled into to following JSON? Your response must be the following JSON with a missing value filled. No other text.
{row_json}

{last_step}"""

        response = call_llm(prompt)
        response_json = parse_str_to_json(response)
        header = self.shared_results["header"]

        # Add updated values to the shared response json
        for colname, val in response_json.items():
            if colname not in header:
                continue

            idx = header.index(colname)
            async with self.lock:
                self.shared_results[self.sample_id][idx] = val


    async def execute_tool(self, page: Page, tool_name: str, element_id: int, value: str):
        """Execute the tool returned by the LLM."""
        selector = f'[data-agent-id="{element_id}"]'

        if tool_name == "type":
            await self.safe_type(page, selector, value)

        elif tool_name == "click":
            await self.safe_click(page, selector)

        elif tool_name == "scroll":
            await self.safe_scroll(page)

        elif tool_name == "screenshot":
            await self.take_and_save_screenshot(self.sample_directory, page, self.num_screenshots)
            self.num_screenshots += 1

        elif tool_name == "finish":
            if self.shared_results:
                await self.write_to_shared_results()
            return True

        else:
            raise ValueError(f"Invalid tool: {tool_name}")

        return False
    

    async def check_if_at_bottom_of_page(self, page: Page) -> bool:
        return await page.evaluate("""
        () => {
            const scrollTop = window.scrollY;
            const viewport = window.innerHeight;
            const fullHeight = document.documentElement.scrollHeight;

            return scrollTop + viewport >= fullHeight - 5;
        }
        """)


    async def run_step(self, page: Page, goal: str, steps_taken: str, has_multiple_objectives: bool) -> tuple[str, str, str | list[int], str]:
        """Run a single step of the agent's interaction with the page, including getting the page state,
        prompting the LLM for the next tool, and parsing the response."""
        img_b64, elements_str = await self.get_page_state(page)

        prev_tool_name = self.history[-1].get("tool_name") if self.history else None
        at_bottom_of_page = await self.check_if_at_bottom_of_page(page)

        tool_name, reasoning, element_id, text = call_llm_tool_call(
            goal, steps_taken, elements_str, img_b64, has_multiple_objectives, prev_tool_name, at_bottom_of_page
        )

        self.history.append({
            "url": page.url,
            "reasoning": reasoning,
            "tool_name": tool_name,
            "element_id": element_id,
            "text": text,
        })

        self.save_history_to_txt()

        return tool_name, element_id, text, reasoning, img_b64

    
    def create_objectives_str(self, objectives: list[str] | None):
        """Create a string representation of the remaining objectives for LLM input."""
        if not objectives:
            return ""
        
        objectives_str = ""
        for i, objective in enumerate(objectives, start=1):
            if objective is not None:
                objectives_str += f"\n {i}) {objective}"

        return f"\n\nThese are the remaining 'Objectives' you must complete:{objectives_str}"
    

    def create_goal_str(self, user_prompt: str, row_info: str | None, objectives: list[str] | None):
        """Create the full goal string for LLM input, including the user prompt, remaining objectives, and any row-specific instructions."""
        objectives_str = self.create_objectives_str(objectives)

        if row_info:
            row_info = f"\n\nYour task is to run the plan for ONLY the following row in the table:\n{row_info}"

        goal = f"Here is the full plan:\n{user_prompt}{objectives_str}{row_info or ''}"
    
        print(f"{self.sample_id} Goal: {goal}")
        return goal
    

    async def mark_objectives_as_completed(self, reasoning: str, objectives: list[str], steps_taken: str, img_b64: bytes) -> bool:
        """Mark the completed objectives as None in the objectives list. If all objectives are completed, return True."""

        objectives_str = self.create_objectives_str(objectives)
        prompt = f"""Which objectives from the following list do you have an answer for based on the 'Steps Taken'? Format response as a valid python list eg. ["1", "2"].{objectives_str}\n\n{steps_taken}"""
        
        # Decide which objectives have been completed
        response = call_llm(prompt, img_b64)
        objective_nums = parse_str_to_list(response) or []

        print(f"{self.sample_id} Completed objectives {objective_nums}: {reasoning}")

        if self.shared_results:
            await self.write_to_shared_results()

        for num in objective_nums:
            objectives[int(num) - 1] = None

        if all(objective is None for objective in objectives):
            return True   # Last objective completed

        return False
    

    async def save_shared_results_to_csv(self) -> None:
        """
        Convert a dict-like JSON structure to CSV.
        
        Rules:
        - 'header' row goes first
        - Remaining keys are sorted alphabetically
        - Each key corresponds to one row
        """
        header = self.shared_results["header"]

        # Get all other keys except 'header'
        other_keys = [k for k in self.shared_results.keys() if k != "header"]

        async with self.lock:
            with open(self.output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                # Write remaining rows
                for key in other_keys:
                    writer.writerow(self.shared_results[key])
    

    async def run_task(self, user_prompt: str, objectives: list[str] | None, row_info: str | None, starting_url: str) -> None:
        """Main function to run the agentic browser task, including navigating to the starting URL, iteratively prompting the LLM for tools, executing those tools, and saving results."""
        # Create an empty output json to save to CSV
        if self.shared_results:
            async with self.lock:
                num_cols = len(self.shared_results["header"])
                self.shared_results[self.sample_id] = [None] * num_cols

        page = await self.context.new_page()
        await page.goto(starting_url, timeout=60000)
        await page.evaluate("document.body.style.zoom = '0.9'")

        # Create instructions (goal) for task
        goal = self.create_goal_str(user_prompt, row_info, objectives)
        has_multiple_objectives = objectives and len(objectives) > 1

        for step in range(self.max_steps):
            try:
                steps_taken = self.convert_history_to_str()

                tool_name, element_id, text, reasoning, img_b64 = await self.run_step(page, goal, steps_taken, has_multiple_objectives)

                # For multi-objective tasks, remove objectives from the list as they are completed
                if tool_name == "objective_completed":
                    done = await self.mark_objectives_as_completed(reasoning, objectives, steps_taken, img_b64)
                    if done:
                        break
                    goal = self.create_goal_str(user_prompt, row_info, objectives)
                else:
                    done = await self.execute_tool(page, tool_name, element_id, text)

            except Exception as e:
                print(f"{self.sample_id} Step {step} failed: {e}")
                continue

            if done:
                break

            await asyncio.sleep(1)

        await self.save_shared_results_to_csv()
        print(f"{self.sample_id} Objective completed.")
