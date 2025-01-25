# core_agent.py

import os
import sys
import json
import subprocess
import tiktoken
from dotenv import load_dotenv

# Import Annotated with fallback:
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from typing import List, Dict
from collections.abc import Callable
import base64  # For base64 encoding
import shlex   # For safely parsing command-line options
from datetime import datetime
import re      # For basic forbidden substring checks

# Load environment variables (for OPENAI_API_KEY, etc.)
load_dotenv()

# We now import the OpenAI class and instantiate it directly:
import openai
from openai import OpenAI

# Provide your API key if not in environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Instantiate two separate clients:
# - client_main: for your primary gpt-4o or similar calls
# - client_summarizer: for summarizing function outputs (e.g. a lesser model in this case gpt-4o-mini)
client_main = OpenAI(
    api_key=openai.api_key
)
client_summarizer = OpenAI(
    api_key=openai.api_key
)

# Initialize tokenizer (be sure this matches the model's tokenizer name)
tokenizer = tiktoken.encoding_for_model("gpt-4o")

# Paths and directories
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_dir)

metadata_path = os.path.join(agent_dir, 'metadata')
prompts_path = os.path.join(agent_dir, 'prompts')
logs_path = os.path.join(metadata_path, 'logs')
sessions_path = os.path.join(metadata_path, 'sessions')

os.makedirs(metadata_path, exist_ok=True)
os.makedirs(prompts_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)
os.makedirs(sessions_path, exist_ok=True)

# EXCEPTIONS AND BASIC HELPERS
class StopException(Exception):
    """Stop Execution by raising this exception (Signal that the task is Finished)."""

def finish(answer: Annotated[str, "Final response to the user."]) -> None:
    """Finish the task with a final answer."""
    raise StopException(answer)

# Global variable to hold target directory
target_directory = ""  # Will store the target directory path

########################################################################
# FUNCTION DECLARATIONS
########################################################################

def list_files(directory: str = "", options: str = "", max_results: int = None) -> str:
    """
    Lists files in the specified directory within the target directory, 
    with optional 'ls' command options and result limit.
    """
    try:
        base_dir = os.path.join(target_directory, directory)
        if not os.path.isdir(base_dir):
            return f"Directory '{directory}' does not exist."

        command = ['ls']
        if options:
            safe_options = shlex.split(options)
            allowed_options = {'-l', '-a', '-h', '-t', '-r', '-S'}
            for opt in safe_options:
                if opt not in allowed_options:
                    return f"Option '{opt}' is not allowed."
            command.extend(safe_options)
        command.append(base_dir)

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error listing files: {result.stderr.strip()}"

        output = result.stdout.strip()

        if max_results is not None:
            lines = output.split('\n')
            output = '\n'.join(lines[:max_results])

        return output
    except Exception as e:
        return f"Error listing files in directory '{directory}': {e}"

def read_file(file_path: str) -> str:
    """
    Reads the content of a file within the target directory, 
    limited to 10000 tokens.
    """
    try:
        full_path = os.path.join(target_directory, file_path)
        with open(full_path, 'r') as f:
            content = f.read()

        # Limit content to 10000 tokens
        max_tokens = 10000
        tokens = tokenizer.encode(content)
        if len(tokens) > max_tokens:
            content = tokenizer.decode(tokens[:max_tokens])
            content += "\n\n[Content truncated due to token limit]"
        return content
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

def search_files(pattern: str, directory: str = "") -> str:
    """
    Searches for files containing the given pattern within the target directory.
    Uses 'grep -ril' to return names of files containing the pattern.
    """
    try:
        base_dir = os.path.join(target_directory, directory)
        if not os.path.isdir(base_dir):
            return f"Directory '{directory}' does not exist."

        command = ['grep', '-ril', pattern, base_dir]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip() or "No files found containing the pattern."
        else:
            return "No files found containing the pattern."
    except Exception as e:
        return f"Error searching files: {e}"

def analyze_image(image_path: str, instruction: str = "") -> str:
    """
    Analyzes an image using the main model's image analysis capabilities (gpt-4o).
    Encodes the image in base64 and sends it to the model with an optional instruction.
    """
    try:
        full_image_path = os.path.join(target_directory, image_path)
        with open(full_image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Base64 encode the image
        base64_image = base64.b64encode(image_data).decode('utf-8')

        content = []
        if instruction:
            content.append({"type": "text", "text": instruction})
        else:
            content.append({"type": "text", "text": "Analyze the following image."})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })

        # We'll send this data in a single user message
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        response = client_main.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.5,
        )
        analysis = response.choices[0].message.content.strip()
        return analysis
    except Exception as e:
        return f"Error analyzing image '{image_path}': {e}"

def run_command(command: str) -> str:
    """
    Executes a shell command with these safeguards:
      1. Disallows forbidden operations (rm -rf, sudo, mv, etc.).
      2. Ensures all file/directory paths are within the target directory 
         (including subdirectories).
    Returns stdout on success or an error message on failure.
    """
    # 1. Forbid certain high-risk strings
    forbidden_substrings = [
        "rm -rf",
        "sudo",
        "mv ",
        "shutdown",
        "reboot",
        "mkfs",
        "fdisk",
        "dd "
    ]
    lowered_command = command.lower()
    for forbidden in forbidden_substrings:
        if forbidden in lowered_command:
            return f"Error: The command contains a forbidden operation: '{forbidden}'"

    # 2. Parse the command. Check all path-like arguments for safe location.
    parsed_command = shlex.split(command)
    target_abs = os.path.abspath(target_directory)

    # We'll iterate over each token to see if it could be a path
    # (if it has a slash, or is not obviously just a flag or subcommand).
    for idx, token in enumerate(parsed_command):
        # Skip if it's the first token (command name) and has no slash
        if idx == 0 and "/" not in token:
            continue

        # Skip if it's obviously just a flag (starts with '-')
        if token.startswith('-'):
            continue

        # We'll interpret any token that has a slash OR might be a path as a path and check it.
        # Attempt to resolve it. If it's a relative path, treat it as relative to target_directory.
        # If it's absolute, we check directly.
        if "/" in token or os.path.exists(token):
            if os.path.isabs(token):
                real_path = os.path.abspath(token)
            else:
                # relative to target directory
                real_path = os.path.abspath(os.path.join(target_directory, token))

            # Check if real_path is still within the target directory
            if not real_path.startswith(target_abs):
                return (f"Error: The command includes a path outside "
                        f"the target directory:\n'{token}' => '{real_path}'")

    # If all checks pass, run the command from within target_directory
    try:
        # By default, let's run the command with cwd set to target_directory
        result = subprocess.run(
            parsed_command,
            capture_output=True,
            text=True,
            cwd=target_directory
        )
        if result.returncode != 0:
            return f"Command error: {result.stderr.strip()}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error running command '{command}': {e}"

def get_available_functions() -> List[Dict[str, str]]:
    """Returns a list of available functions with their descriptions."""
    functions_info = []
    for func_name, func in name_to_function_map.items():
        if func_name != 'get_available_functions':  # Avoid recursion
            desc = func.__doc__ or "No description available."
            functions_info.append({"name": func_name, "description": desc})
    return functions_info

########################################################################
# FUNCTION MAP + SCHEMA GENERATION
########################################################################

name_to_function_map: Dict[str, Callable] = {
    'get_available_functions': get_available_functions,
    'list_files': list_files,
    'read_file': read_file,
    'search_files': search_files,
    'analyze_image': analyze_image,
    'run_command': run_command,
    'finish': finish,
}

def generate_function_schemas():
    schemas = []
    for func_name, func in name_to_function_map.items():
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        if func_name == 'list_files':
            parameters["properties"]["directory"] = {"type": "string", "description": "Subdirectory to list files."}
            parameters["properties"]["options"] = {"type": "string", "description": "Allowed ls options."}
            parameters["properties"]["max_results"] = {"type": "integer", "description": "Max # of results."}
        elif func_name == 'read_file':
            parameters["properties"]["file_path"] = {"type": "string", "description": "Path to the file."}
            parameters["required"].append("file_path")
        elif func_name == 'search_files':
            parameters["properties"]["pattern"] = {"type": "string", "description": "Pattern to grep for."}
            parameters["properties"]["directory"] = {"type": "string", "description": "Subdirectory."}
            parameters["required"].append("pattern")
        elif func_name == 'analyze_image':
            parameters["properties"]["image_path"] = {"type": "string", "description": "Path to the image file."}
            parameters["properties"]["instruction"] = {"type": "string", "description": "Custom instruction."}
            parameters["required"].append("image_path")
        elif func_name == 'run_command':
            parameters["properties"]["command"] = {"type": "string", "description": "The command to execute."}
            parameters["required"].append("command")
        elif func_name == 'finish':
            parameters["properties"]["answer"] = {"type": "string", "description": "Final response."}
            parameters["required"].append("answer")

        function_schema = {
            "name": func_name,
            "description": func.__doc__,
            "parameters": parameters
        }
        schemas.append(function_schema)
    return schemas

########################################################################
# LOGGING / SUMMARIZING HELPERS
########################################################################

def calculate_total_tokens(messages, function_schemas):
    total_tokens = 0
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, list):
            for item in content:
                if 'text' in item:
                    total_tokens += len(tokenizer.encode(item['text']))
                elif 'image_url' in item:
                    total_tokens += len(tokenizer.encode(item['image_url']['url']))
        else:
            total_tokens += len(tokenizer.encode(str(content)))
    # Add function schemas
    functions_str = json.dumps(function_schemas)
    total_tokens += len(tokenizer.encode(functions_str))
    return total_tokens

def summarize_function_output(function_name: str, full_output: str) -> str:
    """
    Uses the 'client_summarizer' to generate a brief summary of the function output.
    """
    try:
        summarizer_messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarizing assistant. You receive raw data from a function call "
                    "and should produce a very short summary. Aim for 1-3 sentences."
                )
            },
            {
                "role": "user",
                "content": f"Function name: {function_name}\nFunction output:\n{full_output}"
            }
        ]
        response = client_summarizer.chat.completions.create(
            model="gpt-4o-mini",
            messages=summarizer_messages,
            max_tokens=200,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return f"[Summary Error: {e}]"

def summarize_interaction(user_request: str, agent_answer: str) -> str:
    """Uses the main gpt-4o model to summarize user-agent interaction in ~1 paragraph."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes interactions."},
        {
            "role": "user",
            "content": (
                f"User requested: {user_request}\n\n"
                f"Agent's answer: {agent_answer}\n\n"
                "Please provide a concise summary of the user request, tasks performed, and final answer."
            )
        }
    ]
    response = client_main.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=4000,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def get_session_title(log_content: str) -> str:
    """Use gpt-4o to create a short 2-5 word name for the session."""
    messages = [
        {"role": "system", "content": "You are an assistant that generates short session titles."},
        {
            "role": "user",
            "content": (
                "Below is the entire log of the user-agent session. "
                "Please provide a short 2-5 word name (with underscores) describing this session.\n\n"
                f"Session Log:\n{log_content}\n\n"
                "Return only the short title."
            )
        }
    ]
    response = client_main.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=20,
        temperature=0.5,
    )
    session_name = response.choices[0].message.content.strip()

    # Basic cleanup
    words = session_name.split()
    words = words[:5]
    if len(words) < 2:
        words.append("Session")
    final_title = "_".join(words)
    return final_title

def log_interaction_to_file(log_message: str):
    """Writes a single line to the log file in logs_path with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_filename = os.path.join(logs_path, "agent_raw.log")
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {log_message}\n")

def save_session_interactions_as_json(session_interactions: List[dict], session_filename: str):
    """Saves session_interactions in a JSON file in sessions folder."""
    output_path = os.path.join(sessions_path, session_filename)
    data_to_write = {"session_interactions": session_interactions}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_write, f, indent=2)

def summarize_log_entry(log_entry: dict) -> str:
    """Sends a log_entry dict to gpt-4o and asks for a <= 3 sentence summary."""
    entry_json = json.dumps(log_entry, indent=2)
    prompt = (
        "The following is a log entry for an AI-based agent.\n"
        "Summarize it into no more than 3 sentences:\n\n"
        f"{entry_json}\n\nSummary:"
    )
    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client_main.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing log entry: {e}"

########################################################################
# MAIN LOOP
########################################################################

def run():
    global target_directory

    session_interactions = []
    interaction_counter = 1

    system_message = {
        "role": "system",
        "content": (
            "You are an AI agent that interacts with the file system using command-line tools.\n\n"
            "You have the following functions available:\n"
            "- list_files(directory: str, options: str, max_results: int)\n"
            "- read_file(file_path: str)\n"
            "- search_files(pattern: str, directory: str)\n"
            "- analyze_image(image_path: str, instruction: str)\n"
            "- run_command(command: str)\n"
            "- finish(answer: str)\n\n"
            "When you need to perform a function, return a JSON with 'function_call' specifying the name & arguments.\n"
            "Continue reasoning until you call finish.\n"
        )
    }

    messages = [system_message]

    # Prompt user for target directory
    desktop_path = os.path.expanduser("~/Desktop")
    print(f"Your Desktop directory is: {desktop_path}")
    target_subdir = input("Please enter the subdirectory within Desktop to use as the target directory: ").strip()
    target_directory = os.path.join(desktop_path, target_subdir)

    if not os.path.exists(target_directory):
        print(f"The directory '{target_directory}' does not exist. Exiting.")
        return
    else:
        print(f"Target directory set to: {target_directory}")
        log_interaction_to_file(f"Target directory set to: {target_directory}")

    function_schemas = generate_function_schemas()

    while True:
        user_input = input("Please enter your instruction (or type 'exit' to end the session): ")
        if user_input.lower() == 'exit':
            print("Ending the session.")
            log_interaction_to_file("User ended the session with 'exit'.")
            break

        # Add user's new instruction to messages
        messages.append({"role": "user", "content": user_input})
        log_interaction_to_file(f"User input: {user_input}")

        max_iterations = 10
        iteration = 0
        while iteration < max_iterations:
            try:
                total_tokens = calculate_total_tokens(messages, function_schemas)
                if total_tokens > 120000:
                    # prune older messages, keep system + last few
                    messages = [messages[0]] + messages[-5:]
                    total_tokens = calculate_total_tokens(messages, function_schemas)

                # Request the assistant's next response
                response = client_main.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    functions=function_schemas,
                    function_call="auto",
                )
                assistant_message = response.choices[0].message

                function_call = assistant_message.function_call
                assistant_content = assistant_message.content or ''

                # If there's plain text
                if assistant_content.strip():
                    print(f"Assistant: {assistant_content}")
                    log_interaction_to_file(f"Assistant said: {assistant_content}")

                    messages.append({"role": "assistant", "content": assistant_content})

                    log_entry = {
                        "interaction_number": interaction_counter,
                        "user_instructions": user_input,
                        "action_type": "text_response",
                        "action_details": None,
                        "action_result": assistant_content
                    }
                    log_entry["notes"] = summarize_log_entry(log_entry)
                    session_interactions.append(log_entry)
                    interaction_counter += 1

                # If there's a function call
                if function_call:
                    function_name = function_call.name
                    if function_name not in name_to_function_map:
                        error_msg = f"Invalid function name: {function_name}"
                        print(error_msg)
                        log_interaction_to_file(error_msg)
                        messages.append({"role": "assistant", "content": error_msg})
                        iteration += 1
                        continue

                    function_to_call = name_to_function_map[function_name]
                    try:
                        function_args = json.loads(function_call.arguments)
                    except json.JSONDecodeError as e:
                        error_msg = f"Error parsing function arguments: {e}"
                        print(error_msg)
                        log_interaction_to_file(error_msg)
                        messages.append({"role": "assistant", "content": error_msg})
                        iteration += 1
                        continue

                    print(f"Calling function '{function_name}' with args: {function_args}")
                    log_interaction_to_file(f"Assistant calls function '{function_name}' with args: {function_args}")

                    try:
                        function_response = function_to_call(**function_args)
                        function_response_str = str(function_response)

                        # Summarize large output
                        summary_of_output = summarize_function_output(function_name, function_response_str)

                        # Log the full output
                        log_interaction_to_file(f"Full function output: {function_response_str}")

                        # Provide only the summary to the conversation
                        messages.append({
                            "role": "function",
                            "name": function_name,
                            "content": summary_of_output
                        })

                        # Build log entry
                        log_entry = {
                            "interaction_number": interaction_counter,
                            "user_instructions": user_input,
                            "action_type": function_name,
                            "action_details": function_args,
                            "action_result": function_response_str
                        }
                        log_entry["notes"] = summarize_log_entry(log_entry)
                        session_interactions.append(log_entry)
                        interaction_counter += 1

                    except StopException as e:
                        # Agent called finish()
                        final_answer = str(e)
                        print(f"Agent finished with message: {final_answer}")
                        log_interaction_to_file(f"Agent finished with message: {final_answer}")

                        summary = summarize_interaction(user_input, final_answer)
                        log_interaction_to_file(f"Summary: {summary}")

                        log_entry = {
                            "interaction_number": interaction_counter,
                            "user_instructions": user_input,
                            "action_type": "finish",
                            "action_details": None,
                            "action_result": final_answer
                        }
                        log_entry["notes"] = summarize_log_entry(log_entry)
                        session_interactions.append(log_entry)
                        interaction_counter += 1

                        # Add final answer to conversation
                        messages.append({"role": "assistant", "content": final_answer})
                        break
                    except Exception as e:
                        err_msg = f"Error calling function '{function_name}': {e}"
                        print(err_msg)
                        log_interaction_to_file(err_msg)
                        messages.append({"role": "assistant", "content": err_msg})
                        iteration += 1
                        continue

                    iteration += 1
                else:
                    # No function call => plain text, break so user can type next instruction
                    iteration += 1
                    break

            except Exception as e:
                err_msg = f"An error occurred in the agent loop: {e}"
                print(err_msg)
                log_interaction_to_file(err_msg)
                break

        print("Session ended for this instruction.")

    # After 'exit', save session logs
    session_log_content = json.dumps(session_interactions, indent=2)
    session_title = get_session_title(session_log_content)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = f"{timestamp_str}_{session_title}.json"
    save_session_interactions_as_json(session_interactions, session_filename)
    log_interaction_to_file(f"Session interactions saved to {session_filename}")

def main():
    print("Starting the agent...")
    run()
    print("Agent has completed the process.")

if __name__ == "__main__":
    main()
