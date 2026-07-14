# Test Optimiser Agent

## Purpose
Optimises a project's test suite by generating and refining tests.

## Framework
LangGraph

## Tools
- `read_tests`: Reads test files from the repo
- `detect_flaky_tests`: Flags tests that fail intermittently

## Workflow
Generate tests, validate coverage, and loop back up to 3 times if the coverage
floor is not met. A human approves the final change.

## State
- `project_id` (str): The project identifier
- `coverage` (float): Current test coverage ratio
- `gen_retry_count` (int): Number of generation attempts so far
- `audit_log` (list): Append-only audit trail
- `tool_errors` (list): Append-only tool error log

## Configuration
Uses OPENAI_API_KEY and a temperature of 0.2.

## Dependencies
langgraph, langchain-openai
