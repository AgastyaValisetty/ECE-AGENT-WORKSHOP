# Project README

## Table of Contents
1. [Overview](#overview)  
2. [Setup](#setup)  
3. [Running the Project](#running-the-project)  
4. [API Reference (botAPI)](#api-reference-botapi)  
5. [Contributing](#contributing)  
6. [License](#license)  

---

## Overview
This repository contains a Python-based toolchain that utilizes **uv** for environment management and **botAPI** for interacting with the Claude Code ecosystem. The project is designed to be easy to navigate, clean, and professional.

---

## Setup
> **Important:** The following steps will create an isolated virtual environment and install all required dependencies.

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Install Python 3.12.6 using uv
uv python install 3.12.6

# 3. Create a virtual environment with Python 3.12.6
uv venv --python 3.12.6

# 4. Activate the virtual environment
#    - Windows PowerShell
.venv\Scripts\activate

# 5. Install project dependencies
uv pip install -r rquirements.txt

# 6. (Optional) Fix PowerShell execution policy if needed
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 7. Reactivate the virtual environment (helps when PowerShell blocks execution)
.venv\Scripts\activate
```

### Why the double activation?
If your PowerShell profile restricts script execution, the second `activate` ensures the environment is properly loaded and usable.

---

## Running the Project
After completing the setup, you can run the main application:

```bash
python -m bot  # Example entry point; adjust as needed
```

Refer to the **Usage** section in the wiki for detailed command‑line options.

---

## API Reference (botAPI)
The `botAPI` module provides a clean interface for interacting with Claude Code and related services. Below is a concise reference of its public functions.

| Function | Description | Signature |
|----------|-------------|-----------|
| `connect()` | Establishes a connection to the Claude Code backend. | `connect(api_key: str = None) -> Connection` |
| `send_message(message: str, connection: Connection) -> str` | Sends a message through the established connection. | `send_message(message: str, connection: Connection) -> str` |
| `listen(connection: Connection, timeout: int = 30) -> str` | Blocks until a response is received or timeout expires. | `listen(connection: Connection, timeout: int = 30) -> str` |
| `disconnect(connection: Connection)` | Gracefully closes the connection. | `disconnect(connection: Connection) -> None` |
| `fetch_status(connection: Connection) -> dict` | Retrieves current status information from the backend. | `fetch_status(connection: Connection) -> dict` |
| `get_bot_config(connection: Connection) -> dict` | Returns the bot’s configuration settings. | `get_bot_config(connection: Connection) -> dict` |

### Detailed Function Docs
- **`connect`**  
  - **Parameters**: `api_key` (optional; if omitted, the system will attempt to use the stored credentials).  
  - **Returns**: A `Connection` object that encapsulates the HTTP session and authentication tokens.  
  - **Example**:  
    ```python
    conn = connect()
    ```

- **`send_message`**  
  - **Parameters**: `message` (the payload to send), `connection` (the `Connection` object).  
  - **Returns**: The raw response string from the backend.  
  - **Example**:  
    ```python
    reply = send_message("Hello, Claude!", conn)
    ```

- **`listen`**  
  - **Parameters**: `connection` (the `Connection` object), `timeout` (seconds before giving up).  
  - **Returns**: The response string or raises a timeout exception.  
  - **Example**:  
    ```python
    response = listen(conn)
    ```

- **`disconnect`**  
  - **Parameters**: `connection` (the `Connection` object).  
  - **Returns**: `None`; closes the underlying session.  
  - **Example**:  
    ```python
    disconnect(conn)
    ```

- **`fetch_status`**  
  - **Parameters**: `connection` (the `Connection` object).  
  - **Returns**: A dictionary containing status flags, version info, and health metrics.  
  - **Example**:  
    ```python
    status = fetch_status(conn)
    ```

- **`get_bot_config`**  
  - **Parameters**: `connection` (the `Connection` object).  
  - **Returns**: Configuration details such as model version, region, and feature flags.  
  - **Example**:  
    ```python
    config = get_bot_config(conn)
    ```

> **Note:** All functions raise a `BotAPIError` on failure; wrap calls in `try/except` blocks for robust error handling.

---

## Contributing
Contributions are welcome! Please follow the standard fork‑branch‑pull‑request workflow.

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feat/your-feature`).  
3. Commit your changes (`git commit -m "Add your feature"`).  
4. Push to the branch (`git push origin feat/your-feature`).  
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

--- 

*Happy hacking!* 🚀