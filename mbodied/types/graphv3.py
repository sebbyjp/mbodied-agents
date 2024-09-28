import asyncio
from pathlib import Path
from typing import Set, Optional, List, Union, Dict
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Callable, Type, TypeVar, Generic, Dict, Tuple, Union

@dataclass
class Parameter:
    name: str
    type: Any
    default: Any = None
    resolved: bool = False
    citation: Optional[Dict[str, str]] = None
    """Citation indicating where to resolve the parameter from in shared_state."""

@dataclass
class Function:
    name: str
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    function_object: Optional[Callable[..., Any]] = None

@dataclass
class ToolCall:
    function: Function
    arguments: Dict[str, Union[Any, Parameter]] = field(default_factory=dict)

class FakeLanguageAgent:
    """A fake LanguageAgent that returns predefined decompositions for testing."""
    
    async def act_and_parse(self, query: str) -> List[Union[str, ToolCall]]:
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Define predefined decompositions across four layers
        decompositions = {
            # Layer 1
            "pick up the marker and place in bin": ["pick up marker", "place in bin"],
            
            # Layer 2
            "pick up marker": ["approach marker", "lower hand", "grasp marker", "lift marker", "retreat"],
            "place in bin": ["approach bin", "lower hand", "release marker", "lift hand", "retreat"],
            
            # Layer 3
            "approach marker": ["get_marker_pose", "move_eef_to_pose"],
            "lower hand": ["move_eef_to_pose"],
            "grasp marker": ["grasp_object"],
            "lift marker": ["move_eef_to_pose"],
            "retreat": ["move_eef_to_pose"],
            "approach bin": ["get_bin_pose", "move_eef_to_pose"],
            "release marker": ["release_object"],
            "lift hand": ["move_eef_to_pose"],
            
            # Layer 4
            "get_marker_pose": [
                ToolCall(
                    function=Function(
                        name="get_object_pose",
                        parameters={
                            "name": Parameter(
                                name="name",
                                type=str,
                                citation={
                                    "key": "world",
                                    "quote": "marker"
                                }
                            )
                        }
                    ),
                    arguments={
                        "name": Parameter(
                            name="name",
                            type=str,
                            citation={
                                "key": "world",
                                "quote": "marker"
                            }
                        )
                    }
                )
            ],
            "get_bin_pose": [
                ToolCall(
                    function=Function(
                        name="get_object_pose",
                        parameters={
                            "name": Parameter(
                                name="name",
                                type=str,
                                citation={
                                    "key": "world",
                                    "quote": "bin"
                                }
                            )
                        }
                    ),
                    arguments={
                        "name": Parameter(
                            name="name",
                            type=str,
                            citation={
                                "key": "world",
                                "quote": "bin"
                            }
                        )
                    }
                )
            ],
            "move_eef_to_pose": [
                ToolCall(
                    function=Function(
                        name="move_eef_pose",
                        parameters={
                            "pose": Parameter(
                                name="pose",
                                type=dict,
                                citation={
                                    "key": "instruction",
                                    "quote": "move eef"
                                }
                            )
                        }
                    ),
                    arguments={
                        "pose": Parameter(
                            name="pose",
                            type=dict,
                            citation={
                                "key": "instruction",
                                "quote": "move eef"
                            }
                        )
                    }
                )
            ],
            "grasp_object": [
                ToolCall(
                    function=Function(
                        name="grasp_object",
                        parameters={
                            "object_pose": Parameter(
                                name="object_pose",
                                type=dict,
                                citation={
                                    "key": "world",
                                    "quote": "marker_pose"
                                }
                            )
                        }
                    ),
                    arguments={
                        "object_pose": Parameter(
                            name="object_pose",
                            type=dict,
                            citation={
                                "key": "world",
                                "quote": "marker_pose"
                            }
                        )
                    }
                )
            ],
            "release_object": [
                ToolCall(
                    function=Function(
                        name="release_object",
                        parameters={
                            "object_pose": Parameter(
                                name="object_pose",
                                type=dict,
                                citation={
                                    "key": "world",
                                    "quote": "marker_pose"
                                }
                            )
                        }
                    ),
                    arguments={
                        "object_pose": Parameter(
                            name="object_pose",
                            type=dict,
                            citation={
                                "key": "world",
                                "quote": "marker_pose"
                            }
                        )
                    }
                )
            ],
        }
        
        # Normalize query to lowercase for matching
        query_normalized = query.lower()
        
        # Return the decomposition if it exists, else return a default ToolCall
        return decompositions.get(query_normalized, [
            ToolCall(
                function=Function(name='ask_human', parameters={}),
                arguments={}
            )
        ])

A = TypeVar("A")
class Assistant(Generic[A]):
    """Assistant that recursively decomposes tasks and executes them asynchronously."""
    agent_class: Type[A]
    def __init__(
        self,
        instruction: str,
        world: str,
        tools: str,
    ):
        """Initialize the assistant with an instruction, world state, and tools.
        
        Args:
            instruction (str): The high-level instruction.
            world (str): The world state as a string.
            tools (str): Function definitions as a string.
            agent_class (Optional[Type[A]]): The LanguageAgent class to use.
        """
        self.shared_state = {
            "world": world,
            "instruction": instruction,
            "tools": tools
        }
        self.task_graph = nx.DiGraph()
        self.responses: List[Dict[str, Any]] = []
        self.task_dict = {}
        self.agent = type(self).agent_class() if type(self).agent_class else FakeLanguageAgent()
        self.num_tasks = -1
        self.tasks = []
        self.tree = {}

    def __class_getitem__(cls, agent_class: Type[A]) -> Type["Assistant[A]"]:
      cls.agent_class = agent_class
      return cls

    async def run(self):
        """Starts the task decomposition process."""
        # Start with Layer 1
        decomposition = await self.agent.act_and_parse(self.shared_state["instruction"])
        for task in decomposition:
            await self.process_task(task, parent=None, task_dict=self.task_dict)
    
    async def process_task(self, task: Union[str, ToolCall], parent: Optional[str], task_dict=None  ):
        """Processes a single task, adding it to the task graph and decomposing further if necessary.
        
        Args:
            task (Union[str, ToolCall]): The task to process.
            parent (Optional[str]): The parent task's name.
        """
        self.num_tasks += 1
        self.tasks.append(task)
        self.tree[parent] = str(task)
        if isinstance(task, str):
            # Decompose further
            subtasks = await self.agent.act_and_parse(task)
            current_task_name = f"{task}_{self.num_tasks}"
            self.task_graph.add_node(current_task_name, data={"status": "pending"})
            if parent:
                self.task_graph.add_edge(parent, current_task_name)
            if task_dict:
                task_dict[current_task_name] = task
            for subtask in subtasks:
                await self.process_task(subtask, parent=current_task_name)
        elif isinstance(task, ToolCall):
            # Add ToolCall to task graph
            task_name = f"{task.function.name}_{self.num_tasks}"
            self.task_graph.add_node(task_name, data={"status": "pending", "tool_call": task})
            if parent:
                self.task_graph.add_edge(parent, task_name)
            if task_dict:
                task_dict[task_name] = task
            # Check for unresolved parameters
            for arg_name, arg_value in task.arguments.items():
                if isinstance(arg_value, Parameter) and not arg_value.resolved:
                    # Create a parameter resolution task
                    param_task_name = f"resolve_{arg_value.name}_{self.num_tasks}"
                    self.task_graph.add_node(param_task_name, data={"status": "pending", "parameter": arg_value})
                    self.task_graph.add_edge(param_task_name, task_name)
    
    async def execute_tasks(self):
        """Executes tasks in the task graph."""
        while not self.all_tasks_completed():
            for node in list(self.task_graph.nodes):
                data = self.task_graph.nodes[node]['data']
                if data['status'] == 'pending':
                    if 'parameter' in data:
                        # Resolve parameter
                        parameter: Parameter = data['parameter']
                        citation = parameter.citation
                        if citation:
                            key = citation["key"]
                            quote = citation["quote"]
                            # Fetch the value from shared_state
                            if key in self.shared_state:
                                # For simplicity, assume 'world' is a string in "object:pose" format
                                if key == "world":
                                    # Parse the world string to extract the needed pose or object
                                    objects = self.shared_state["world"].split(", ")
                                    obj_dict = {}
                                    for obj in objects:
                                        obj_name, obj_pose = obj.split(":pose(")
                                        obj_pose = obj_pose.rstrip(")")
                                        obj_dict[obj_name.strip()] = obj_pose
                                    resolved_value = obj_dict.get(quote, parameter.default)
                                else:
                                    resolved_value = self.shared_state.get(key, parameter.default)
                            else:
                                resolved_value = parameter.default
                            
                            # Update the parameter
                            parameter.resolved = True
                            
                            # Update the ToolCall's argument
                            predecessors = list(self.task_graph.predecessors(node))
                            for pred in predecessors:
                                pred_data = self.task_graph.nodes[pred]['data']
                                if 'tool_call' in pred_data:
                                    tool_call: ToolCall = pred_data['tool_call']
                                    if tool_call.arguments.get(parameter.name) == parameter:
                                        tool_call.arguments[parameter.name] = resolved_value
                            
                            # Mark parameter resolution as completed
                            data['status'] = 'completed'
                            print(f"Parameter '{parameter.name}' resolved with value: {resolved_value}")
                    
                    elif 'tool_call' in data:
                        tool_call: ToolCall = data['tool_call']
                        # Check if all parameters are resolved
                        if all(
                            not isinstance(arg, Parameter) or arg.resolved
                            for arg in tool_call.arguments.values()
                        ):
                            # Execute the ToolCall
                            args = {k: v for k, v in tool_call.arguments.items()}
                            print(f"Executing {tool_call.function.name} with arguments {args}")
                            response = f"Executed {tool_call.function.name} with {args}"
                            # Record the response
                            self.responses.append({
                                "tool_call": tool_call.function.name,
                                "response": response
                            })
                            # Mark as completed
                            data['status'] = 'completed'
                            print(f"ToolCall '{tool_call.function.name}' executed successfully.")
            
            await asyncio.sleep(0.1)  # Yield control to the event loop
    
    def all_tasks_completed(self) -> bool:
        """Checks if all tasks in the graph have been completed.
        
        Returns:
            bool: True if all tasks are completed, False otherwise.
        """
        return all(data.get('status') == 'completed' for _, data in self.task_graph.nodes(data=True))
    
    def visualize_graph(self, filename: Optional[str] = None):
        """Visualize the task graph using networkx and matplotlib.
        
        Args:
            filename (Optional[str]): If provided, saves the graph to the given filename.
        """
        plt.figure(figsize=(48, 32))
        pos = nx.planar_layout(nx.topological_sort(self.task_graph), scale=2)
        nx.draw(
            self.task_graph, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', node_size=2000, font_size=10, font_weight='bold'
        )
        
        # Draw edge labels if needed
        # edge_labels = nx.get_edge_attributes(self.task_graph, 'label')
        # nx.draw_networkx_edge_labels(self.task_graph, pos, edge_labels=edge_labels)
        
        plt.title("Task Decomposition Graph")
        plt.axis('off')
        
        if filename:
            plt.savefig(filename)
            print(f"Graph saved as {filename}")
        else:
            plt.show()
    
    def to_markdown(self,*, show=False) -> str:
        """Generates collapsible Markdown representing the task graph.
        
        Returns:
            str: The generated Markdown string.
        """
        markdown_lines = []
        visited: Set[str] = set()
        
        # Identify root nodes (nodes with no predecessors)
        root_nodes = [node for node in self.task_graph.nodes if self.task_graph.in_degree(node) == 0]
        
        for root in root_nodes:
            self._traverse_and_build_markdown(root, markdown_lines, visited, level=0)
        
        md = "\n".join(markdown_lines)

        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        console.print(Markdown(md))
        Path("task_graph.md").write_text(md)
    
    def _traverse_and_build_markdown(self, node: str, markdown_lines: List[str], visited: Set[str], level: int):
        """Recursively traverses the graph to build Markdown.
        
        Args:
            node (str): Current node identifier.
            markdown_lines (List[str]): Accumulates Markdown lines.
            visited (Set[str]): Tracks visited nodes to prevent cycles.
            level (int): Current indentation level.
        """
        def wrap_in_details(summary: str, content: str | None = None) -> str:
            return f"<details><summary>{summary}</summary>\n{content}\n</details>"
        if node in visited:
            return

        if level == 0:
            indent = "# "
        elif level == 1:
            indent = "## "
        else:
            indent = "  " * level
        data = self.task_graph.nodes[node]['data']
        
        if 'tool_call' in data:
            tool_call: ToolCall = data['tool_call']
            display_name = f"{tool_call.function.name}"
            markdown_lines.append(wrap_in_details(display_name, ""))
            
            # Include function arguments with citations
            for arg, param in tool_call.arguments.items():
                if isinstance(param, Parameter):
                    citation = param.citation
                    citation_text = f" *(Cited from `{citation['key']}`: \"{citation['quote']}\")*" if citation else ""
                    markdown_lines.append(f"{indent}  - **{arg}**: `{param.name}` {citation_text}")
                else:
                    markdown_lines.append(f"{indent}  - **{arg}**: `{param}`")
            
            # Check for dependent tasks (children in the graph)
            children = list(self.task_graph.successors(node))
            if children:
                for child in children:
                    self._traverse_and_build_markdown(child, markdown_lines, visited, level=level+1)
            
            markdown_lines.append(f"{indent}</details>")
        else:
            # It's a composite task
            task_name = node
            markdown_lines.append(f"{indent}- <details><summary> {task_name}</summary>")
            
            # Traverse children
            children = list(self.task_graph.successors(node))
            for child in children:
                self._traverse_and_build_markdown(child, markdown_lines, visited, level=level+1)
            
            markdown_lines.append(f"{indent}</details>")


if __name__ == "__main__":
  a = Assistant[FakeLanguageAgent]("Pick up the marker and place in bin", "marker:pose(1, 2, 3), bin:pose(4, 5, 6)", "get_object_pose, move_eef_pose, grasp_object, release_object")
  asyncio.run(a.run())
  a.visualize_graph("task_graph.png")
  a.to_markdown(show=True)