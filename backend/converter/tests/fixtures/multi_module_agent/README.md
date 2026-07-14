# Mini Optimiser Agent

## Purpose
A compact multi-module LangGraph agent used as a conversion fixture.

## Framework
LangGraph

## Tools
- `read_tests`: Reads test files from a path
- `detect_conventions`: Detects suite conventions

## Workflow
Intake, then check the coverage floor and loop back to revise up to a cap, then
generate and validate tests (retry up to 3 times), with human approval steps.

## State
- `project_id` (str): project identifier
- `coverage` (float): current coverage ratio
- `gen_retry_count` (int): generation attempts
- `audit_log` (list): append-only audit trail

## Configuration
Uses GEMINI_API_KEY and MAX_GEN_RETRIES.

## Dependencies
langgraph, langchain-core, google-genai, pytest
