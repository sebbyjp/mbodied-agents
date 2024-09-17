# %% [markdown]
# <a href="https://colab.research.google.com/github/sebbyjp/mbodied-agents/blob/prod/dspy_self_discovery.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# #Self-Discovery Prompting Documentation
# 
# This documentation outlines the operational framework of the Self-Discovery Prompting https://arxiv.org/pdf/2402.03620.pdf, which employs a modular approach to facilitate reasoning and problem-solving tasks. The system integrates several components, each serving a unique role in the overall problem-solving process.
# 
# #Imports

# %%
!pip install yt-dlp
!pip install ffmpeg ffprobe
!pip install openai-whisper
!pip install dspy-ai
!pip install sentence-transformers
!pip install openai pydub

# %% [markdown]
# #Setup

# %%
import dspy
from dspy.teleprompt import BootstrapFewShot
import openai
import json
import pandas as pd
from datasets import load_dataset
import re



openai.api_key = ''

# Configure the language models to use
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
gpt4 = dspy.OpenAI(model='gpt-4-0125-preview',max_tokens=2048)

dspy.settings.configure(lm=gpt4)

# %% [markdown]
# #Reasoning Modules

# %%
# Usage example
reasoning_modules = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this problem and its solutions?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "Is the problem an analytical one that requrires data analysis, modeling, or optimization techniques?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Given the problem specification and the current best solution, have a guess about other possible solutions.",
    "Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    "Let’s think step by step.",
    "Let’s make a step by step plan and implement it with good notion and explanation."
]
seed_modules_str = '"'+ '","'.join(reasoning_modules)  # for a newline-separated string


# %% [markdown]
# #SelectModules
# ###Purpose
# 
# Selects appropriate reasoning modules from a predefined set based on the input task description.
# 
# ###Parameters
# 
# question: The task or question to be solved.
# 
# reasoning_modules: A list of available reasoning modules.
# 
# ###Returns
# 
# A dictionary containing selected modules tailored to the task at hand.

# %%
# Define some example input-output pairs for module selection
module_selection_examples = [
    dspy.Example(
        task_description="How to improve customer satisfaction for an online store?",
        seed_modules=json.dumps(reasoning_modules),
        selected_modules=json.dumps([
            "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
            "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
            "What are the underlying causes or factors contributing to the problem?",
            "What are the potential obstacles or challenges that might arise in solving this problem?"
        ])
    ).with_inputs('task_description', 'seed_modules'),
    dspy.Example(
        task_description="How to optimize the supply chain for a manufacturing company?",
        seed_modules=json.dumps(reasoning_modules),
        selected_modules=json.dumps([
            "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
            "What are the key assumptions underlying this problem?",
            "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
            "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?"
        ])
    ).with_inputs('task_description', 'seed_modules'),
    dspy.Example(
        task_description="How many numbers between 1 and 2005 are integer multiples of 3 or 4 but not 12?",
        seed_modules=json.dumps(reasoning_modules),
        selected_modules=json.dumps([
            "What is the core issue or problem that needs to be addressed?",
            "How can I measure progress on this problem?",
            "What kinds of solution typically are produced for this kind of problem specification?",
            "Given the problem specification and the current best solution, have a guess about other possible solutions.",
            "Let’s think step by step.",
            "Let’s make a step by step plan and implement it with good notion and explanation."
        ])
    ).with_inputs('task_description', 'seed_modules'),
    dspy.Example(
        task_description="How many numbers are in the list 6,7,10,11,14,15,..., 94,95,98?",
        seed_modules=json.dumps(reasoning_modules),
        selected_modules=json.dumps([
            "What is the core issue or problem that needs to be addressed?",
            "How can I simplify the problem so that it is easier to solve?",
            "How can I measure progress on this problem?",
            "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
            "Ignoring the current best solution, create an entirely new solution to the problem.",
            "Let’s think step by step.",
            "Let’s make a step by step plan and implement it with good notion and explanation."
        ])
    ).with_inputs('task_description', 'seed_modules')
]
# Update the signatures to include a mechanism for passing extra context
class SelectModulesSignature(dspy.Signature):
    """"Select several reasoning modules that are crucial to utilize in order to solve the given task, print the complete reasoning module to pass validation"""
    task_description = dspy.InputField(prefix="Task(s) Description:", desc="The task(s) to solve.")
    seed_modules = dspy.InputField(prefix="Relevant Reasoning Modules:", desc="List of relevant reasoning modules to solve task(s) with.")
    selected_modules = dspy.OutputField(prefix="Selected Reasoning Modules (No Hallucinations and write out the whole reasoning module). DO NOT SOLVE THE PROBLEM:", desc="['module_1', 'module_2', ...]")

# Module Selection
class SelectModules(dspy.Module):
    def __init__(self):
        super().__init__()
        self.select_modules = dspy.ChainOfThought(SelectModulesSignature)

    def forward(self, task_description, seed_modules):
        seed_modules_json = json.dumps(seed_modules)
        prediction = self.select_modules(task_description=task_description, seed_modules=seed_modules_json)
        selected_modules = prediction.selected_modules
        # Assert that the selected_modules output is in JSON format

        return dspy.Prediction(selected_modules=selected_modules)


def validate_selected_modules_against_reasoning_modules(example, prediction, something_else):
    # Convert reasoning_modules from JSON string to Python list
    try:
        predicted_selected_modules_list = json.loads(prediction.selected_modules)
    except Exception as e:
        print(f'Failed to load predicted selected modules: {e}')
        return False
    return all(module in reasoning_modules for module in predicted_selected_modules_list)

# Set up a basic teleprompter for module selection
teleprompter_selection = BootstrapFewShot(metric=validate_selected_modules_against_reasoning_modules)

# Compile the SelectModules module
compiled_select_modules = teleprompter_selection.compile(SelectModules(), trainset=module_selection_examples)

# %% [markdown]
# # Test for select modules

# %%
task_description = 'How many primes are in the row of Pascal’s Triangle that starts with a 1 followed by a 6?'
prediction = compiled_select_modules.forward(task_description=task_description, seed_modules=reasoning_modules)['selected_modules']
prediction

# %% [markdown]
# #AdaptModules
# ###Purpose
# 
# Adapts the selected reasoning modules to the specific context and requirements of the task.
# 
# 
# ###Parameters
# 
# selected_modules: The modules selected by the SelectModules component.
# 
# question: The task or question to be solved.
# ###Returns
# 
# A dictionary of adapted modules ready for implementation in solving the task.

# %%
# Define some example input-output pairs for module adaptation
module_adaptation_examples = [
    dspy.Example(
        task_description="How many numbers between 1 and 2005 are integer multiples of 3 or 4 but not 12?",
        selected_modules=json.dumps([
            "What is the core issue or problem that needs to be addressed?",
            "How can I measure progress on this problem?",
            "What kinds of solution typically are produced for this kind of problem specification?",
            "Given the problem specification and the current best solution, have a guess about other possible solutions.",
            "Let’s think step by step.",
            "Let’s make a step by step plan and implement it with good notion and explanation."
        ]),
        adapted_modules=json.dumps([
            "Identify the mathematical pattern or rule that defines the set of numbers.",
            "Determine a method to count the numbers that fit the criteria without enumerating each one.",
            "Consider the inclusion-exclusion principle to avoid overcounting numbers that are multiples of both 3 and 4.",
            "Estimate the number of multiples within the range and adjust for the exclusion of multiples of 12.",
            "Break down the problem into smaller parts: count multiples of 3, multiples of 4, and then exclude multiples of 12.",
            "Implement a systematic approach to calculate the final count, ensuring all criteria are met."
        ])
    ).with_inputs('task_description', 'selected_modules'),
    dspy.Example(
        task_description="A restaurant offers three desserts, and exactly twice as many appetizers as main courses. A dinner consists of an appetizer, a main course, and a dessert. What is the least number of main courses that the restaurant should offer so that a customer could have a different dinner each night in the year 2003?",
        selected_modules=json.dumps([
            "What is the core issue or problem that needs to be addressed?",
            "How can I simplify the problem so that it is easier to solve?",
            "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
            "Let’s think step by step.",
            "Let’s make a step by step plan and implement it with good notion and explanation."
        ]),
        adapted_modules=json.dumps([
            "Determine the total number of dinner combinations needed to cover all nights in the year 2003.",
            "Recognize that the number of dinners is the product of the number of choices for each course.",
            "Formulate the problem as an equation based on the given ratios and the total number of combinations required.",
            "Solve the equation to find the minimum number of main courses needed.",
            "Verify the solution by ensuring that the number of possible dinners exceeds the number of days in the year 2003."
        ])
    ).with_inputs('task_description', 'selected_modules'),
    dspy.Example(
        task_description="How many ways are there to arrange 6 people around a circular table with 7 seats? (Two seatings are considered the same if one is a rotation of the other.)",
        selected_modules=json.dumps([
            "What is the core issue or problem that needs to be addressed?",
            "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
            "Let’s think step by step.",
            "Let’s make a step by step plan and implement it with good notion and explanation."
        ]),
        adapted_modules=json.dumps([
            "Understand the concept of circular permutations and how they differ from linear permutations.",
            "Recognize that one seat can be fixed to account for rotational symmetry, reducing the problem to a linear permutation.",
            "Calculate the number of ways to arrange the remaining people in the available seats.",
            "Consider any constraints or special cases that may affect the arrangement.",
            "Conclude with the total number of distinct circular arrangements for the given scenario."
        ])
    ).with_inputs('task_description', 'selected_modules')
    # Add more examples as needed
]

# Define the signature for module adaptation
class AdaptModulesSignature(dspy.Signature):
    """Rephrase and specify each reasoning module so that it better helps solving the task."""
    selected_modules = dspy.InputField(prefix="Selected Reasoning Modukes:", desc="The selected reasoning modules to use for the task.")
    task_description = dspy.InputField(prefix="Task(s) Description:", desc="The task description.")
    adapted_modules = dspy.OutputField(prefix="Adapted Modules:", desc="['adapted_reasoning_module_1','adapted_reasoning_module_2','adapted_reasoning_module_N']")

# Implement the module for adapting reasoning modules
class AdaptModules(dspy.Module):
    def __init__(self):
        super().__init__()
        self.adapt_modules = dspy.ChainOfThought(AdaptModulesSignature)

    def forward(self, selected_modules, task_description):
        selected_modules_json = json.dumps(selected_modules)
        prediction = self.adapt_modules(selected_modules=selected_modules_json, task_description=task_description)
        adapted_modules = []  # Initialize adapted_modules to an empty list
        try:
            cleaned_output = json.dumps(prediction.adapted_modules.strip())
            adapted_modules = json.loads(cleaned_output)
        except Exception as e:
            print(f'Failed to load adapted modules: {e}')
        return dspy.Prediction(adapted_modules=adapted_modules)

# Validation logic for module adaptation
def validate_adapted_modules(example, pred, trace=None):
    # gold_adapted_modules = json.loads(example.adapted_modules)
    # predicted_adapted_modules = pred.adapted_modules
    return True# set(gold_adapted_modules) == set(predicted_adapted_modules)

# Set up a basic teleprompter for module adaptation
teleprompter_adaptation = BootstrapFewShot(metric=validate_adapted_modules)

# Compile the AdaptModules module
compiled_adapt_modules = teleprompter_adaptation.compile(AdaptModules(), trainset=module_adaptation_examples)

# %% [markdown]
# Test for adapted Modules

# %%
task_description = 'How many primes are in the row of Pascal’s Triangle that starts with a 1 followed by a 6?'
selected_modules = ['What is the core issue or problem that needs to be addressed?',
 'Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?',
 'Let’s think step by step.',
 'Let’s make a step by step plan and implement it with good notion and explanation.']

prediction = compiled_adapt_modules.forward(task_description=task_description, selected_modules=selected_modules)
prediction['adapted_modules']

# %% [markdown]
# #ImplementStructure
# ###Purpose
# 
# Translates the adapted modules into a concrete, step-by-step reasoning structure in JSON format.
# 
# 
# ###Parameters
# 
# adapted_modules: Adapted modules from the AdaptModules component.
# 
# task_description: Detailed description of the task to be solved.
# 
# ###Returns
# 
# A JSON object detailing the step-by-step reasoning structure for solving the task.

# %%
implement_structure_examples = [
    dspy.Example(
        task_description='This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle)',
        adapted_modules=json.dumps([
            "Analyze the SVG path commands and coordinates to understand the shape being drawn.",
            "Visualize the sequence of moves and lines to form a mental image of the resulting shape.",
            "Compare the visualized shape with the characteristics of known geometric shapes to identify a match.",
            "Use pattern recognition and knowledge of geometric properties to interpret the SVG path.",
            "Verify the conclusion by sketching or digitally rendering the SVG path to ensure accuracy."
        ]),
        reasoning_structure=json.dumps({
            "steps": [
                {
                    "step": 1,
                    "description": "Examine the SVG path commands and break them down into individual movements and lines.",
                    "action": "Identify each command in the SVG path and understand its function, such as moving to a new starting point or drawing a line."
                },
                {
                    "step": 2,
                    "description": "Analyze the Move to Command (M) and Line to Command (L) within the SVG path.",
                    "action": "Determine the starting points and the end points of each line segment to visualize the shape being drawn."
                },
                {
                    "step": 3,
                    "description": "Visualize the sequence of moves and lines to form a mental image of the resulting shape.",
                    "action": "Use the coordinates provided by the path commands to plot the points and connect them in the order they appear."
                },
                {
                    "step": 4,
                    "description": "Compare the visualized shape with the characteristics of known geometric shapes to identify a match.",
                    "action": "Assess the number of sides, angles, and other geometric properties to classify the shape."
                },
                {
                    "step": 5,
                    "description": "Use pattern recognition and knowledge of geometric properties to interpret the SVG path.",
                    "action": "Apply knowledge of geometric shapes to determine if the path represents a common geometric figure."
                },
                {
                    "step": 6,
                    "description": "Verify the conclusion by sketching or digitally rendering the SVG path to ensure accuracy.",
                    "action": "Create a visual representation of the path to confirm the shape identified in the previous steps."
                }
            ],
            "conclusion": {
                "description": "Synthesize the information from the analysis to determine the geometric shape represented by the SVG path.",
                "action": "Review the visual representation and the properties of the shape to conclude the type of geometric figure it represents."
            }
        })
    ).with_inputs('task_description', 'adapted_modules'),
    dspy.Example(
        task_description='A customer ordered 15 pieces of gourmet chocolate. The order can be packaged in small boxes that contain 1, 2 or 4 pieces of chocolate. Any box that is used must be full. How many different combinations of boxes can be used for the customer’s 15 chocolate pieces? One such combination to be included is to use seven 2-piece boxes and one 1-piece box.',
        adapted_modules=json.dumps([
            "Identify the constraints on box sizes and the requirement for boxes to be full.",
            "Systematically explore all combinations of box sizes that sum up to 15 pieces of chocolate.",
            "Iterate through possible numbers of 4-piece boxes, adjusting for the remaining pieces.",
            "For each number of 4-piece boxes, iterate through possible numbers of 2-piece boxes, adjusting for the remaining pieces.",
            "Determine if the remaining pieces can be filled with 1-piece boxes for each combination of 4-piece and 2-piece boxes.",
            "Ensure all possible combinations are accounted for without duplication."
        ]),
        reasoning_structure=json.dumps({
            "steps": [
                {
                    "step": 1,
                    "description": "Identify the constraints on box sizes and the requirement for boxes to be full.",
                    "action": "Note that boxes can only contain 1, 2, or 4 pieces of chocolate and must be full."
                },
                {
                    "step": 2,
                    "description": "Systematically explore all combinations of box sizes that sum up to 15 pieces of chocolate.",
                    "action": "Start with combinations involving the largest box size and work down to the smallest."
                },
                {
                    "step": 3,
                    "description": "Iterate through possible numbers of 4-piece boxes, adjusting for the remaining pieces.",
                    "action": "Calculate how many pieces are left after using 0, 1, 2, or 3 boxes of 4 pieces."
                },
                {
                    "step": 4,
                    "description": "For each number of 4-piece boxes, iterate through possible numbers of 2-piece boxes, adjusting for the remaining pieces.",
                    "action": "For each scenario from step 3, calculate how many 2-piece boxes can be used with the remaining pieces."
                },
                {
                    "step": 5,
                    "description": "Determine if the remaining pieces can be filled with 1-piece boxes for each combination of 4-piece and 2-piece boxes.",
                    "action": "Check if the remaining pieces after steps 3 and 4 can be exactly filled with 1-piece boxes."
                },
                {
                    "step": 6,
                    "description": "Ensure all possible combinations are accounted for without duplication.",
                    "action": "Review all generated combinations to ensure no duplicates and all possibilities are considered."
                }
            ],
            "conclusion": {
                "description": "Calculate the total number of unique combinations that can be used to package the 15 pieces of chocolate.",
                "action": "Summarize the findings and present the total number of distinct packaging combinations."
            }
        })
    ).with_inputs('task_description', 'adapted_modules')
]


# Define the signature for implementing the reasoning structure
class ImplementStructureSignature(dspy.Signature):
    """Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format. Only output JSON no other text:"""
    adapted_modules = dspy.InputField(prefix="Adapted Modules:", desc="Adapted Modules")
    task_description = dspy.InputField(prefix="Task(s) Description:", desc="The task description.")
    reasoning_structure = dspy.OutputField(prefix="Reasoning Structure JSON:", desc="Implement a reasoning structure for solvers to follow step-by-step and arrive at the correct answers. just json no other text")

# Implement the module for creating a reasoning structure
class ImplementStructure(dspy.Module):
    def __init__(self):
        super().__init__()
        self.implement_structure = dspy.Predict(ImplementStructureSignature)

    def forward(self, adapted_modules, task_description):
        adapted_modules_json = json.dumps(adapted_modules)
        prediction = self.implement_structure(adapted_modules=adapted_modules_json, task_description=task_description)
        reasoning_structure = {}  # Initialize reasoning_structure to an empty dictionary
        try:
            reasoning_structure_output = prediction.reasoning_structure
            reasoning_structure_cleaned = re.sub(r'^\s*```json\n|\n```$', '', reasoning_structure_output, flags=re.MULTILINE).strip()
            reasoning_structure = json.loads(reasoning_structure_cleaned)
        except Exception as e:
            print(f'Failed to load reasoning structure: {e}')
        return dspy.Prediction(reasoning_structure=reasoning_structure)

# Validation logic for module adaptation
def validate_adapted_modules(example, pred, trace=None):

    return True# set(gold_adapted_modules) == set(predicted_adapted_modules)

# Compile the ImplementStructureSignature using the teleprompter
teleprompter_implement_structure = BootstrapFewShot(metric=validate_adapted_modules)
compiled_implement_structure = teleprompter_implement_structure.compile(ImplementStructure(), trainset=implement_structure_examples)

# %% [markdown]
# Test code

# %%
# Call the compiled ImplementStructure module with example inputs
example_adapted_modules = '["Identify the specific row in Pascal\'s Triangle starting with 1 followed by a 6.", "Understand the definition of prime numbers and the criteria for identifying them.", "Systematically check each number in the row (excluding the 1s) for primality.", "Tally the numbers confirmed to be prime to answer the question."]'

task_description = 'How many primes are in the row of Pascal’s Triangle that starts with a 1 followed by a 6?'

# Generate the reasoning structure
reasoning_structure_prediction = compiled_implement_structure(adapted_modules=example_adapted_modules, task_description=task_description)

# Print the reasoning structure
print(reasoning_structure_prediction.reasoning_structure)


# %% [markdown]
# #Solver
# ###Purpose
# 
# Applies the reasoning structure to generate a solution to the task.
# 
# 
# ###Parameters
# 
# question: The task or question to be solved.
# 
# reasoning_structure: The step-by-step plan generated by the ImplementStructure component.
# ###Returns
# 
# The solution to the task as determined through the application of the reasoning structure.

# %%
solver_examples = [
    dspy.Example(
        task_description="How many numbers between 1 and 2005 are integer multiples of 3 or 4 but not 12?",
        reasoning_structure=json.dumps({
            "steps": [
                {
                    "step": 1,
                    "description": "Identify the mathematical pattern or rule that defines the set of numbers.",
                    "action": "Determine the multiples of 3 and 4 up to 2005, excluding those that are multiples of 12."
                },
                {
                    "step": 2,
                    "description": "Determine a method to count the numbers that fit the criteria without enumerating each one.",
                    "action": "Use the inclusion-exclusion principle to calculate the count."
                },
                {
                    "step": 3,
                    "description": "Estimate the number of multiples within the range and adjust for the exclusion of multiples of 12.",
                    "action": "Subtract the count of multiples of 12 from the combined count of multiples of 3 and 4."
                },
                {
                    "step": 4,
                    "description": "Break down the problem into smaller parts: count multiples of 3, multiples of 4, and then exclude multiples of 12.",
                    "action": "Calculate the counts separately and then combine them according to the inclusion-exclusion principle."
                },
                {
                    "step": 5,
                    "description": "Implement a systematic approach to calculate the final count, ensuring all criteria are met.",
                    "action": "Verify the calculations and ensure no number is counted more than once."
                }
            ],
            "conclusion": {
                "description": "Synthesize the information from the analysis to determine the final count of numbers.",
                "action": "Combine the results from each step to arrive at the final answer."
            }
        }),
        solution=json.dumps({
            "steps": [
                {
                    "step": 1,
                    "answer": "There are 668 multiples of 3 and 501 multiples of 4 between 1 and 2005."
                },
                {
                    "step": 2,
                    "answer": "Using the inclusion-exclusion principle, we can find the count without listing each number."
                },
                {
                    "step": 3,
                    "answer": "There are 167 multiples of 12 between 1 and 2005."
                },
                {
                    "step": 4,
                    "answer": "The combined count of multiples of 3 and 4 is 1169, but we must exclude the multiples of 12."
                },
                {
                    "step": 5,
                    "answer": "After excluding multiples of 12, we are left with 1002 numbers."
                }
            ],
            "conclusion": {
                "answer": "1002"
            }
        })
    ).with_inputs('task_description', 'reasoning_structure'),
]

# Define the signature for implementing the reasoning structure with answers
class SolverSignature(dspy.Signature):
    """Operationalize the reasoning modules into a step-by-step reasoning plan with answers and a conclusion."""
    task_description = dspy.InputField(prefix="Adapted Modules:", desc="The task description.")
    reasoning_structure = dspy.InputField(prefix="Reasoning Structure:", desc="The reasoning structure")
    solution = dspy.OutputField(prefix="Solution:", desc='The complete reasoning plan with answers and a conclusion in JSON format. example:"{steps[{"step": N,"answer": "ASNWER TEXT"}],"conclusion": {"answer": "ONLY ANSWER NO SUPPORTING TEXT"}}')

# Implement the solver module
class Solver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve_task = dspy.Predict(SolverSignature)

    def forward(self, task_description, reasoning_structure):
        reasoning_structure_json = json.dumps(reasoning_structure)
        prediction = self.solve_task(task_description=task_description, reasoning_structure=reasoning_structure_json)
        solution = {}  # Initialize solution to an empty dictionary
        try:
            cleaned_output = re.sub(r'^\s*```json\n|\n```$', '', prediction.solution, flags=re.MULTILINE).strip()
            solution = json.loads(cleaned_output)
        except Exception as e:
            print(f'Failed to load solution: {e}')
        return dspy.Prediction(solution=solution)

# Validation logic for the solver
def validate_solution(example, pred, trace=None):
    # Here you would compare the predicted solution with the expected solution
    # For simplicity, we are returning True
    return True

# Compile the Solver module using the teleprompter
teleprompter_solver = BootstrapFewShot(metric=validate_solution)

# Compile the Solver module
compiled_solver = teleprompter_solver.compile(Solver(), trainset=solver_examples)

# %% [markdown]
# Testing Code

# %%
# Function to call the compiled_solver with a new novel example input without the answers
def call_solver_with_new_input(task_description, reasoning_structure):
    # Convert the reasoning structure to JSON format for the solver input
    reasoning_structure_json = json.dumps(reasoning_structure)

    # Use the compiled solver to predict the solution for the new example input
    prediction = compiled_solver.solve_task(task_description=task_description, reasoning_structure=reasoning_structure_json)

    # Return the predicted solution
    return prediction.solution

# Example of calling the function with a new input
new_example_input = {
    "task_description": "A customer ordered 15 pieces of gourmet chocolate. The order can be packaged in small boxes that contain 1, 2 or 4 pieces of chocolate. Any box that is used must be full. How many different combinations of boxes can be used for the customer’s 15 chocolate pieces? One such combination to be included is to use seven 2-piece boxes and one 1-piece box.",
    "reasoning_structure": {
        "steps": [
            {
                "step": 1,
                "description": "Identify the constraints on box sizes and the requirement for boxes to be full.",
                "action": "Note that boxes can only contain 1, 2, or 4 pieces of chocolate and must be full."
            },
            {
                "step": 2,
                "description": "Systematically explore all combinations of box sizes that sum up to 15 pieces of chocolate.",
                "action": "Start with combinations involving the largest box size and work down to the smallest."
            },
            {
                "step": 3,
                "description": "Iterate through possible numbers of 4-piece boxes, adjusting for the remaining pieces.",
                "action": "Calculate how many pieces are left after using 0, 1, 2, or 3 boxes of 4 pieces."
            },
            {
                "step": 4,
                "description": "For each number of 4-piece boxes, iterate through possible numbers of 2-piece boxes, adjusting for the remaining pieces.",
                "action": "For each scenario from step 3, calculate how many 2-piece boxes can be used with the remaining pieces."
            },
            {
                "step": 5,
                "description": "Determine if the remaining pieces can be filled with 1-piece boxes for each combination of 4-piece and 2-piece boxes.",
                "action": "Check if the remaining pieces after steps 3 and 4 can be exactly filled with 1-piece boxes."
            },
            {
                "step": 6,
                "description": "Ensure all possible combinations are accounted for without duplication.",
                "action": "Review all generated combinations to ensure no duplicates and all possibilities are considered."
            }
        ],
        "conclusion": {
            "description": "Calculate the total number of unique combinations that can be used to package the 15 pieces of chocolate.",
            "action": "Summarize the findings and present the total number of distinct packaging combinations."
        }
    }
}

# Example call to the function
# Note: In a real scenario, the 'answer' fields would not be included in the input
predicted_solution = call_solver_with_new_input(new_example_input['task_description'], new_example_input['reasoning_structure'])

# Print the predicted solution
print("Predicted Solution:", predicted_solution)

# %% [markdown]
# #Eval Against BBH Dataset
# 
# Loading in a random sample of 5 questions from each of the question types

# %%
from datasets import load_dataset
import random

configs = ['tracking_shuffled_objects_seven_objects', 'salient_translation_error_detection', 'tracking_shuffled_objects_three_objects', 'geometric_shapes', 'object_counting', 'word_sorting', 'logical_deduction_five_objects', 'hyperbaton', 'sports_understanding', 'logical_deduction_seven_objects', 'multistep_arithmetic_two', 'ruin_names', 'causal_judgement', 'logical_deduction_three_objects', 'formal_fallacies', 'snarks', 'boolean_expressions', 'reasoning_about_colored_objects', 'dyck_languages', 'navigate', 'disambiguation_qa', 'temporal_sequences', 'web_of_lies', 'tracking_shuffled_objects_five_objects', 'penguins_in_a_table', 'movie_recommendation', 'date_understanding']

eval_examples = []
for config in configs:
    dataset = load_dataset("maveriq/bigbenchhard", config)["train"]
    sampled_records = random.sample(list(dataset), 5)  # Randomly sample 5 records from the dataset
    for r in sampled_records:
        eval_examples.append(dspy.Example({"question": r["input"], "answer": r["target"]}).with_inputs("question"))
print(f"Total examples collected: {len(eval_examples)}")


# %% [markdown]
# Using LLMs as an evaluator... would not recommend

# %%

class LLMAnswerEvaluationSignature(dspy.Signature):
    """Evaluate whether the predicted answer is given the gold answer"""

    question = dspy.InputField(prefix="Question:", desc="The question.")
    reasoning_structure = dspy.InputField(prefix="Reasoning Structure:", desc="The reasoning structure that explains the answers")
    predicted_answer = dspy.InputField(prefix="Predicted Answer:", desc="The predicted answer.")
    gold_answer = dspy.InputField(prefix="Gold Answer:", desc="The gold answer.")
    is_correct = dspy.OutputField(prefix="Is Correct:", desc="True or False")

class LLMAnswerEvaluation(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate_answer = dspy.Predict(LLMAnswerEvaluationSignature)

    def forward(self, question, reasoning_structure, predicted_answer, gold_answer):
        return self.evaluate_answer(
            question=question, reasoning_structure=reasoning_structure, predicted_answer=predicted_answer, gold_answer=gold_answer
        )

# Examples for compilation (with outputs added)
examples = [
    dspy.Example(
        question="On the desk, there are three turquoise plates, one pink plate, and three pink booklets. If I remove all the pink things from the desk, how many booklets remain on it? Options: (A) zero (B) one (C) two (D) three (E) four (F) five (G) six (H) seven (I) eight (J) nine (K) ten (L) eleven (M) twelve (N) thirteen (O) fourteen (P) fifteen (Q) sixteen",
        reasoning_structure="",
        predicted_answer="After removing all pink items from the desk, the number of booklets remaining is zero.",
        gold_answer="(A)",
        is_correct="True",  # Output added
    ).with_inputs("question", "reasoning_structure", "predicted_answer", "gold_answer"),
    dspy.Example(
        question="On the desk, you see a fuchsia dog leash and a teal necklace. Is the dog leash turquoise? Options: (A) yes (B) no",
        reasoning_structure="",
        predicted_answer="The dog leash is not turquoise",
        gold_answer="(B)",
        is_correct="True",  # Output added
    ).with_inputs("question", "reasoning_structure", "predicted_answer", "gold_answer"),
    dspy.Example(
        question="This SVG path element <path d=\"M 22.00,62.00 L 46.00,65.00 L 64.00,60.00 L 91.00,42.00 L 92.00,24.00 L 46.00,19.00 L 22.00,62.00\"/> draws a\nOptions:\n(A) circle\n(B) heptagon\n(C) hexagon\n(D) kite\n(E) line\n(F) octagon\n(G) pentagon\n(H) rectangle\n(I) sector\n(J) triangle:",
        reasoning_structure="",
        predicted_answer="The SVG path element draws a hexagon",
        gold_answer="(C)",
        is_correct="True",
    ).with_inputs("question", "reasoning_structure", "predicted_answer", "gold_answer"),
    dspy.Example(
        question="Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: ( ( ( [ < [ < [ ( [ { < ( < ( ( ) ) { } [ ] > ) > } ] ) { < > } ] [ < [ { ( < < { { [ < { [ [ < > [ ] ] ] } > ] { { } } } } > > ) } ] > { < > { } } ] > ( [ ] ) ] > ] ) ( ):",
        reasoning_structure="",
        predicted_answer="The completed sequence is '(((((<[<[([<{<((()){}}[]>}>}])<{<>}>]<[{(<{{{{[<{[[<>[]]]}>]{{}}}}>>)}]>]{<>{}}]>)[[]])>))'",
        gold_answer=")",
        is_correct="True",
    ).with_inputs("question", "reasoning_structure", "predicted_answer", "gold_answer"),
    dspy.Example(
        question="Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the date one week ago from today in MM/DD/YYYY? \Options:\n(A) 03/03/2017\n(B) 04/27/2017\n(C) 03/21/2017\n(D) 02/21/1961\n(E) 02/21/2017:",
        reasoning_structure="",
        predicted_answer="E",
        gold_answer="(E)",
        is_correct="True"
    ).with_inputs("question",  "reasoning_structure", "predicted_answer", "gold_answer"),
    dspy.Example(
        question="Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: ( ) ( (:",
        reasoning_structure="",
        predicted_answer="( ) ( () )",  # You can replace this with the predicted answer from your model
        gold_answer=" ) )",
        is_correct="True"
    ).with_inputs("question", "reasoning_structure", "predicted_answer", "gold_answer")
]

# Validation logic for the solver (updated to check for correctness)
def validate_solution(example, pred, trace=None):
    # Check if the predicted 'is_correct' matches the expected 'is_correct'
    return pred.is_correct.lower() == example.is_correct.lower()

# Compile the LLMAnswerEvaluation module using the teleprompter
teleprompter_solver = BootstrapFewShot(metric=validate_solution, max_bootstrapped_demos=6)
compiled_answer_evaluator = teleprompter_solver.compile(LLMAnswerEvaluation(), trainset=examples)

# %% [markdown]
# Put it all together and then run the eval

# %%
from dspy.evaluate import Evaluate


def eval_metric(true, prediction, trace=None):
    true_answer = true.answer
    reasoning_structure = str(prediction['steps'])
    predicted_answer = prediction['conclusion']['answer']

    is_correct_pred = compiled_answer_evaluator(
        question=true.question, reasoning_structure=reasoning_structure,predicted_answer=predicted_answer, gold_answer=true_answer
    )
    if is_correct_pred.is_correct.lower() == "true":
        return True
    else:
        print(f'evaluation failed for {true.question}:')
        print(f'{predicted_answer} against {true_answer}')
        return False

class CombinedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.select_modules = SelectModules()
        self.adapt_modules = AdaptModules()
        self.implement_structure = ImplementStructure()
        self.solver = Solver()

    def forward(self, question):
        selected_modules = self.select_modules(question, reasoning_modules).selected_modules
        adapted_modules = self.adapt_modules(selected_modules, question).adapted_modules
        reasoning_structure = self.implement_structure(adapted_modules, question).reasoning_structure
        solution = self.solver(question, reasoning_structure).solution
        return solution

combined_module = CombinedModule()


# Evaluation code (same as before)
evaluate = Evaluate(devset=eval_examples, metric=eval_metric, num_threads=25, display_progress=True, display_table=10)
evaluate(combined_module)


