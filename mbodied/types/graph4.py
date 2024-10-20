import asyncio
from mbodied.types.graphv3 import Assistant, FakeLanguageAgent, ToolCall, Function, Parameter
# Sample shared_state setup
shared_state = {
    "world": "marker:pose(marker_pose), bin:pose(bin_pose)",
    "instruction": "pick up the marker and place in bin",
    "tools": "get_object_pose(name: str), move_eef_pose(pose: dict), grasp_object(object_pose: dict), release_object(object_pose: dict)"
}

# Initialize the Assistant
assistant = Assistant(
    instruction=shared_state["instruction"],
    world=shared_state["world"],
    tools=shared_state["tools"],
    agent_class=FakeLanguageAgent
)

# Manually build the task graph for demonstration
graph = assistant.task_graph

# Layer 1
graph.add_node("pick up the marker and place in bin", data={"status": "pending"})
graph.add_node("pick up marker", data={"status": "pending"})
graph.add_node("place in bin", data={"status": "pending"})
graph.add_edge("pick up the marker and place in bin", "pick up marker")
graph.add_edge("pick up the marker and place in bin", "place in bin")

# Layer 2 - Subtasks for "pick up marker"
graph.add_node("approach marker", data={"status": "pending"})
graph.add_node("lower hand", data={"status": "pending"})
graph.add_node("grasp marker", data={"status": "pending"})
graph.add_node("lift marker", data={"status": "pending"})
graph.add_node("retreat", data={"status": "pending"})
graph.add_edge("pick up marker", "approach marker")
graph.add_edge("pick up marker", "lower hand")
graph.add_edge("pick up marker", "grasp marker")
graph.add_edge("pick up marker", "lift marker")
graph.add_edge("pick up marker", "retreat")

# Layer 2 - Subtasks for "place in bin"
graph.add_node("approach bin", data={"status": "pending"})
graph.add_node("lower hand", data={"status": "pending"})
graph.add_node("release marker", data={"status": "pending"})
graph.add_node("lift hand", data={"status": "pending"})
graph.add_node("retreat", data={"status": "pending"})
graph.add_edge("place in bin", "approach bin")
graph.add_edge("place in bin", "lower hand")
graph.add_edge("place in bin", "release marker")
graph.add_edge("place in bin", "lift hand")
graph.add_edge("place in bin", "retreat")

# Layer 3 - ToolCalls for "approach marker"
get_marker_pose_call = ToolCall(
    function=Function(
        name="get_object_pose",
        parameters={
            "name": Parameter(
                name="name",
                type=str,
                citation={"key": "world", "quote": "marker"}
            )
        }
    ),
    arguments={
        "name": Parameter(
            name="name",
            type=str,
            citation={"key": "world", "quote": "marker"}
        )
    }
)

move_eef_to_pose_call = ToolCall(
    function=Function(
        name="move_eef_pose",
        parameters={
            "pose": Parameter(
                name="pose",
                type=dict,
                citation={"key": "instruction", "quote": "move eef"}
            )
        }
    ),
    arguments={
        "pose": Parameter(
            name="pose",
            type=dict,
            citation={"key": "instruction", "quote": "move eef"}
        )
    }
)

graph.add_node("get_marker_pose", data={"status": "pending", "tool_call": get_marker_pose_call})
graph.add_node("move_eef_to_pose", data={"status": "pending", "tool_call": move_eef_to_pose_call})
graph.add_edge("approach marker", "get_marker_pose")
graph.add_edge("approach marker", "move_eef_to_pose")

# Similarly, add ToolCalls for other Layer 3 tasks
# For brevity, we'll assume that all Layer 3 tasks have their corresponding ToolCalls added similarly.

# Run the Assistant
async def run_assistant():
    await assistant.execute_tasks()
    # Generate and print the collapsible markdown
    markdown = assistant.generate_collapsible_markdown()
    print("### Collapsible Markdown Output:\n")
    print(markdown)

# Execute the asynchronous function
asyncio.run(run_assistant())
