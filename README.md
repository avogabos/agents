# Core Agent README

Welcome to **Core Agent**, an AI-driven command-line utility that demonstrates how to integrate GPT-based functionality with local system actions. This README will guide you through:

- [Overview](#overview)  
- [Features and Workflow](#features-and-workflow)  
- [Prompts and Message Flow](#prompts-and-message-flow)  
- [Function Descriptions](#function-descriptions)  
- [Environment Setup](#environment-setup)  
- [Running the Agent](#running-the-agent)  
- [Session Logging](#session-logging)  
- [How It Works (Detailed Explanation)](#how-it-works-detailed-explanation)  
- [FAQ](#faq)

---

## Overview

**Core Agent** is a Python script that:

1. Uses the OpenAI API (or a similar GPT-based service) to interpret user instructions in a conversational loop.  
2. Can call local functions—like listing files, reading files, searching file contents, analyzing images, or running commands—based on GPT’s responses.  
3. Logs interactions and function outputs, summarizing them to keep the conversation’s token usage manageable.

By combining GPT’s language generation with Python’s ability to interact with the filesystem, **Core Agent** can act as a powerful (yet safe-guarded) assistant to help you perform tasks programmatically while maintaining a rich conversational interface.

---

## Features and Workflow

1. **Sets a Target Directory**  
   - At launch, you choose a target directory (typically a subdirectory on your Desktop). All file operations are restricted to this folder for safety.

2. **Interactive Conversation Loop**  
   - You type instructions in the terminal.  
   - The AI (GPT) responds either with a direct text response or requests to call a local function for tasks like listing files or reading file content.

3. **Safety Checks**  
   - The script disallows dangerous operations (e.g., `rm -rf`, `sudo`, `shutdown`, etc.).  
   - Path arguments are validated to ensure they stay within the target directory.

4. **Summaries**  
   - The system uses a secondary summarizing model (`client_summarizer`) to condense large outputs so the conversation remains within token limits.

5. **Logging and Session Files**  
   - A running log of all actions is kept in `metadata/logs/agent_raw.log`.  
   - At the end of each session, a JSON file capturing the entire conversation and function calls is saved in `metadata/sessions/`.

---

## Prompts and Message Flow

Internally, **Core Agent** maintains a Python list called `messages`. It appends new messages for each:

- **System**: The initial instruction that sets the overall context for GPT.  
- **User**: Anything you (the user) type in.  
- **Assistant**: GPT’s textual responses.  
- **Function**: Summaries of function outputs.

At each user instruction, the script sends the entire `messages` list (including the system prompt and the conversation so far) to GPT. If GPT decides a function call is needed, it returns a JSON structure (e.g., `{"name": "list_files", "arguments": {...}}`). The script then executes that function, logs the result, and puts a *summarized* version of that result back into `messages` as a role `"function"` message. GPT sees that summary on the next iteration, preserving context without ballooning token usage.

**High-Level Flow**:

1. **User Input** → 2. **GPT** → 3. **Optional Function Call** → 4. **Summarize Output** → 5. **Assistant** → (repeat)

---

## Function Descriptions

**Core Agent** defines and maps these functions, which GPT can call by name:

- **`list_files(directory, options, max_results)`**  
  Lists files in a subdirectory of the target folder with optional `ls` flags and a result limit.
  
- **`read_file(file_path)`**  
  Reads a file’s contents (up to 10,000 tokens) from the target folder.

- **`search_files(pattern, directory)`**  
  Performs a grep-based search (`-ril`) for a text pattern in the specified subdirectory.

- **`analyze_image(image_path, instruction)`**  
  Base64-encodes an image, sends it to the main GPT model, and returns analysis (e.g., describing the image).

- **`run_command(command)`**  
  Executes a shell command restricted to the target directory, disallowing destructive or root-level commands.

- **`finish(answer)`**  
  Ends the conversation with a final answer to the user.

- **`get_available_functions()`**  
  Returns a list of available function names and their docstrings.

---

## Environment Setup

1. **Install Dependencies**  
   - Python 3.8+  
   - `pip install openai python-dotenv tiktoken typing_extensions`

2. **Set OpenAI API Key**  
   - Create a `.env` file containing  
     ```bash
     OPENAI_API_KEY=YOUR_OPENAI_KEY
     ```  
     Or export it in your shell environment.

3. **File Structure**  
   This script expects certain directories to exist (and creates them if they don’t):
   ```
   core_agent/
   ├─ metadata/
   │  ├─ logs/
   │  └─ sessions/
   ├─ prompts/
   └─ core_agent.py
   ```

4. **(Optional) GPT Model Configuration**  
   - By default, it references an imaginary “gpt-4o” and “gpt-4o-mini.” Replace these with actual OpenAI model names if needed (e.g., `"gpt-3.5-turbo"`) or your own model endpoints.

---

## Running the Agent

1. **Clone or Place the Script**  
   - Put `core_agent.py` in a directory of your choice (ensuring dependencies are available).

2. **Execute**  
   ```bash
   python core_agent.py
   ```
   - The script starts by printing “Starting the agent...”

3. **Choose Target Directory**  
   - You’ll be asked:  
     ```
     Your Desktop directory is: /home/username/Desktop
     Please enter the subdirectory within Desktop to use as the target directory:
     ```
   - Enter something like `my_project`.

4. **Interact**  
   - After setting the target directory, type instructions one by one. Examples:
     - `List files in this directory with -l option`
     - `Read file "notes.txt"`
     - `Search for the pattern "TODO" in subfolder "src"`
   - The AI will respond, possibly calling the relevant function. You’ll see either text replies or function call logs.

5. **End the Session**  
   - Type `exit` to finish. 
   - The script saves a JSON session log in `metadata/sessions/`, with a GPT-generated short title.

---

## Session Logging

**Core Agent** creates two types of logs:

1. **agent_raw.log**  
   - A text file in `metadata/logs/`.  
   - Records every user input, GPT message, and function call in a timestamped manner.

2. **Session JSON**  
   - On each session exit, a JSON file is saved to `metadata/sessions/`.  
   - Contains an array of `session_interactions`, each representing a user message, GPT text response, or function call.  
   - The script uses GPT to generate a concise session title (e.g., `YYYYMMDD_HHMMSS_Short_Title.json`).

---

## How It Works (Detailed Explanation)

### 1. Initialization

- **`main()`** → Calls **`run()`**.  
- **Environment**: Loads `.env` and sets up two OpenAI clients (`client_main` and `client_summarizer`).  
- **Paths**: Creates/fetches `metadata/`, `prompts/`, `logs/`, and `sessions/` directories.

### 2. Conversation Setup

- A **system message** is constructed to instruct GPT about its purpose and the available functions.  
- An empty list `messages` starts with just this system message.

### 3. Target Directory Selection

- The user is asked for a subdirectory on their Desktop. This becomes the restricted `target_directory`.

### 4. Main Loop

**While** `True`:
1. Prompt the user for an instruction.  
2. If user types `exit`, break. Otherwise, append `{"role": "user", "content": user_input}` to `messages`.
3. Enter an **internal** loop (up to `max_iterations`) to let GPT respond *multiple times* until it’s done with the user’s request.  

### 5. GPT Message Creation

Each iteration:

1. **Calculate tokens** in `messages` to ensure we don’t exceed a large threshold. If it’s too big, prune older messages.  
2. **Call** `client_main.chat.completions.create(...)` with:
   - The entire `messages` list.
   - The dynamically generated **function schemas** (so GPT knows how to call them).
   - `function_call="auto"` so GPT can choose to call functions when needed.

3. **Parse** GPT’s response:
   - If it has plain text (`assistant_message.content`), print it and log it.  
   - If it requests a **function call** (`assistant_message.function_call`), do the following:
     - Parse the arguments.
     - Execute the corresponding Python function from `name_to_function_map`.
     - Summarize the raw output via `client_summarizer`.
     - Append the summary to `messages` with `role="function"`.
   - If the function called is `finish(answer)`, raise a `StopException` to end the conversation with that final answer.

### 6. Summaries

- Any large output from a function is summarized before appending to `messages`.  
- Additionally, each log entry is summarized for permanent storage in `session_interactions`.

### 7. Session End

- When the user types `exit` (or GPT calls `finish()`):
  - The code logs the last messages, uses GPT to generate a short session title, and writes `session_interactions` to a JSON file in `metadata/sessions/`.

---

## FAQ

**1. Why Summaries?**  
Large outputs (like file listings or file contents) might exceed token limits. Summaries keep the conversation relevant without including the entire raw data.

**2. How Do I Add My Own Functions?**  
Add a new function in the code, document it, and add it to the `name_to_function_map`. Also modify `generate_function_schemas` to describe its parameters.

**3. Is This Secure?**  
The script disallows certain commands (`rm -rf`, `sudo`, etc.) and ensures paths are under a user-specified directory. Still, it’s a proof-of-concept. Always audit code and outputs before production usage.

**4. Can I Use a Different Model?**  
Yes. Edit the references to `"gpt-4o"` or `"gpt-4o-mini"` with your preferred model (e.g., `"gpt-3.5-turbo"` or `"gpt-4"`).  

---

## Conclusion

**Core Agent** is a foundational example of how to combine GPT’s interactive conversation abilities with local function calls—offering a flexible yet managed way to execute filesystem tasks, get logs, and maintain context. Feel free to adapt it further for your own workflows, add more specialized functions, or integrate advanced logging and security measures.

Enjoy exploring and extending **Core Agent**!