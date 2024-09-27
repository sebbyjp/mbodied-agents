import asyncio
import time
from collections import ChainMap, defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Generic, List, Optional, Type, TypeVar, Union, Generator, Set

import networkx as nx
import unittest

# Type variables
A = TypeVar("A", bound="LanguageAgent")

# Global context variable to manage context across assistants
context_var = ContextVar("context_var", default=ChainMap())

# Shared state to be updated globally
shared_state = {"world": None, "human_instruction": None, "current_hand_pose": None}


# Mocked WorldObject and Pose
@dataclass
class Pose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    reference_frame: str
    origin: str

    def __repr__(self):
        return (
            f"Pose(x={self.x}, y={self.y}, z={self.z}, roll={self.roll}, "
            f"pitch={self.pitch}, yaw={self.yaw}, reference_frame='{self.reference_frame}', origin='{self.origin}')"
        )


@dataclass
class WorldObject:
    name: str
    pose: Pose

    def __repr__(self):
        return f"WorldObject(name='{self.name}', pose={self.pose})"


# Mocked MultiDict to represent the world state
class MultiDict(defaultdict):
    def __init__(self):
        super().__init__(list)

    def add(self, key: str, value: WorldObject):
        self[key].append(value)


# Mocked LanguageAgent class with basic functionality
class LanguageAgent:
    """Mocked LanguageAgent class with basic functionality."""

    async def act(self, query: str, context: Optional[str] = None) -> str:
        # Simulate agent response with a slight delay
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Agent response to '{query}' with context '{context}'"

    async def act_and_parse(self, query: str) -> List[str]:
        # Simulate parsing with a slight delay
        await asyncio.sleep(0.1)
        # For testing purposes, we'll recursively split the query into two subtasks until a base case
        if len(query.split()) <= 2:
            return [query]  # Base case: return the query as is
        else:
            mid = len(query.split()) // 2
            subtask1 = " ".join(query.split()[:mid])
            subtask2 = " ".join(query.split()[mid:])
            return [subtask1, subtask2]


# Context manager for child context
@contextmanager
def child_context(context_var: ContextVar[ChainMap], context: Dict[Any, Any]):
    current_context = context_var.get()
    new_context = current_context.new_child(context)
    token: Token = context_var.set(new_context)
    try:
        yield
    finally:
        context_var.reset(token)


@dataclass
class Task:
    name: str
    conditions: Dict[str, Any] = field(default_factory=dict)  # e.g., {'python_version': '>=3.6'}
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    incompatibilities: List[Set[str]] = field(default_factory=list)  # Incompatible task combinations


# Assistant class
class Assistant(Generic[A]):
    """Assistant that recursively decomposes tasks and executes them asynchronously."""

    def __init__(
        self,
        instruction: str,
        world: MultiDict[str, WorldObject],
        current_hand_pose: Pose,
        agent_class: Optional[Type[A]] = None,
    ):
        """Initialize the assistant with an instruction and world state."""
        self.instruction = instruction
        self.world = world
        self.current_hand_pose = current_hand_pose
        self.agent_class = agent_class or LanguageAgent

        # Shared state (to be updated automatically)
        self.shared_state = shared_state
        self.shared_state["world"] = self.world
        self.shared_state["human_instruction"] = self.instruction
        self.shared_state["current_hand_pose"] = self.current_hand_pose

        # Initialize the task graph
        self.task_graph = nx.DiGraph()
        self.responses: Deque[Dict[str, Any]] = deque()

        # Initialize agents
        self.consider_agent: A = self.agent_class()
        self.judge_agent: A = self.agent_class()
        self.decompose_agent: A = self.agent_class()
        self.coordinator_agent: A = self.agent_class()

        # Parse instruction and build the task graph
        asyncio.run(self.parse_instruction())

    async def parse_instruction(self):
        """Recursively parse the instruction and build the task graph."""
        await self.recursive_parse(self.instruction, parent_task=None)

    async def recursive_parse(self, instruction: str, parent_task: Optional[str] = None):
        """Recursively decompose instructions into subtasks."""
        # Call LanguageAgent to decompose the instruction
        subtasks = await self.decompose_agent.act_and_parse(instruction)

        # Base case: if no further decomposition, add the instruction as a task
        if len(subtasks) == 1 and subtasks[0] == instruction:
            task_name = instruction.strip()
            self.task_graph.add_node(task_name, data={"status": "pending"})
            if parent_task:
                self.task_graph.add_edge(parent_task, task_name)
            return

        # Otherwise, recursively parse each subtask
        current_task = instruction.strip()
        self.task_graph.add_node(current_task, data={"status": "pending"})
        if parent_task:
            self.task_graph.add_edge(parent_task, current_task)

        for subtask in subtasks:
            await self.recursive_parse(subtask, parent_task=current_task)

    async def execute_tasks(self):
        """Execute tasks in the task graph asynchronously without blocking."""
        task_futures = {}
        execution_plan_generator = self.generate_execution_plan()

        async for step in self._async_generator_wrapper(execution_plan_generator):
            if isinstance(step, list):
                # Schedule concurrent tasks
                tasks = [self.schedule_task(task_name, task_futures) for task_name in step]
                await asyncio.gather(*tasks)
            else:
                # Schedule single task
                await self.schedule_task(step, task_futures)

    async def _async_generator_wrapper(self, generator):
        """Wrap a synchronous generator to make it asynchronous."""
        for item in generator:
            yield item
            await asyncio.sleep(0)  # Yield control to the event loop

    async def schedule_task(self, task_name: str, task_futures: Dict[str, asyncio.Task]):
        """Schedule a task for execution without blocking."""
        # Check if task is already scheduled
        if task_name in task_futures:
            return await task_futures[task_name]

        # Schedule task
        task_future = asyncio.create_task(self.execute_task(task_name))
        task_futures[task_name] = task_future

        # Wait for the task to complete
        return await task_future

    async def execute_task(self, task_name: str):
        """Execute an individual task asynchronously."""
        # Check if task is already completed
        if self.task_graph.nodes[task_name]["data"]["status"] == "completed":
            return

        # Simulate task execution
        print(f"Executing task: {task_name}")
        task_context = self.get_task_context(task_name)
        # For recursive tasks, we can consider whether to decompose further
        # For simplicity, we'll assume tasks at this point are atomic
        response = await self.consider_agent.act(f"Execute {task_name}", context=task_context)
        # Mark task as completed
        self.task_graph.nodes[task_name]["data"]["status"] = "completed"
        self.task_graph.nodes[task_name]["data"]["response"] = response
        # Update responses deque asynchronously
        self.responses.append(
            {
                "last_response": response,
                "shared_state": self.shared_state.copy(),
                "human_instruction": self.instruction,
                "completed": task_name,
            }
        )

    def get_task_context(self, task_name: str) -> str:
        """Generate context for a task based on the shared state."""
        # Serialize the shared state into a string
        context = f"Instruction: {self.shared_state['human_instruction']}\n"
        context += f"Current Hand Pose: {self.shared_state['current_hand_pose']}\n"
        context += "World Objects:\n"
        for objs in self.shared_state["world"].values():
            for obj in objs:
                context += f"- {obj.name} at {obj.pose}\n"
        return context

    async def run(self):
        """Main execution loop running at 10 Hz."""
        interval = 0.1  # 10 Hz
        while not self.all_tasks_completed():
            start_time = time.time()
            await self.execute_tasks()
            elapsed = time.time() - start_time
            time_to_wait = max(0, interval - elapsed)
            await asyncio.sleep(time_to_wait)
            # For demonstration, we can simulate updates to the world or instruction here if needed

    def all_tasks_completed(self) -> bool:
        """Check if all tasks have been completed."""
        return all(data["data"].get("status") == "completed" for _, data in self.task_graph.nodes(data=True))

    def get_responses(self) -> List[Dict[str, Any]]:
        """Get the list of responses."""
        return list(self.responses)

    def generate_execution_plan(self) -> Generator[Union[str, List[str]], None, None]:
        """Generate a sequence of function calls for execution.

        Yields each task or list of tasks as they become ready for execution.
        """
        from collections import deque

        # Copy of the task graph to manipulate
        graph = self.task_graph.copy()
        execution_plan = []

        # Initialize in-degree for each node
        in_degree = {node: graph.in_degree(node) for node in graph.nodes()}

        # Queue of tasks ready to be executed (in-degree zero)
        ready_tasks = deque([node for node in graph.nodes() if in_degree[node] == 0])

        while ready_tasks:
            # Tasks that can be executed in parallel at this stage
            concurrent_tasks = []

            # Collect all tasks that are ready (in-degree zero)
            initial_ready_tasks = list(ready_tasks)
            ready_tasks.clear()
            for task in initial_ready_tasks:
                concurrent_tasks.append(task)

            # Yield the tasks
            if len(concurrent_tasks) == 1:
                yield concurrent_tasks[0]
            else:
                yield concurrent_tasks

            # Update in-degree of successor tasks
            for task in concurrent_tasks:
                for successor in graph.successors(task):
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        ready_tasks.append(successor)

            # Remove executed tasks from the graph
            graph.remove_nodes_from(concurrent_tasks)

    # Additional methods for conditions and conflicts can be added here as needed


# Test cases for the Assistant class
class TestAssistant(unittest.TestCase):
    def setUp(self):
        # Create a mock world state
        self.world = MultiDict[str, WorldObject]()
        pose1 = Pose(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, "map", "origin")
        world_object1 = WorldObject(name="cube1", pose=pose1)
        self.world.add("cube", world_object1)

        pose2 = Pose(3.0, 4.0, 0.0, 0.0, 0.0, 0.0, "map", "origin")
        world_object2 = WorldObject(name="cube2", pose=pose2)
        self.world.add("cube", world_object2)

        # Current hand pose
        self.current_hand_pose = Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "map", "origin")

        self.instruction = "Pick up all cubes and place them on the table."

        self.assistant = Assistant[LanguageAgent](
            instruction=self.instruction,
            world=self.world,
            current_hand_pose=self.current_hand_pose,
            agent_class=LanguageAgent,
        )

    def test_recursive_decomposition(self):
        # Test that the assistant correctly builds a recursive task graph
        async def run_assistant():
            await self.assistant.run()

        asyncio.run(run_assistant())

        # Check if all tasks are completed
        self.assertTrue(self.assistant.all_tasks_completed())

        # Print the task graph
        print("\nTask Graph:")
        for node in self.assistant.task_graph.nodes:
            print(node)

        # Print the responses
        responses = self.assistant.get_responses()
        print("\nResponses:")
        for response in responses:
            print(response)

        # Verify that the number of tasks scales appropriately
        # For the given instruction, the expected depth depends on the length of the instruction
        # Since we split the instruction into halves each time, the number of tasks should be O(log n)
        # For this test, we'll just check that there are multiple tasks
        self.assertGreater(len(self.assistant.task_graph.nodes), 1)


# Main function to run the tests
def run_tests():
    unittest.main()


if __name__ == "__main__":
    run_tests()
