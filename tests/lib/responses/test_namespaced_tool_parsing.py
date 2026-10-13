from __future__ import annotations

import json
from typing import Any

import pytest
import pydantic

import openai
from openai import OpenAI
from tests.respx2 import MockRouter
from openai.lib._parsing._responses import get_input_tool_by_name

from ...conftest import base_url


class CrmLookup(pydantic.BaseModel):
    customer_id: str


class CatalogLookup(pydantic.BaseModel):
    sku: int


def _crm_namespace_tool() -> dict[str, Any]:
    return {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [openai.pydantic_responses_function_tool(CrmLookup, name="lookup")],
    }


def _catalog_top_level_tool() -> Any:
    return openai.pydantic_responses_function_tool(CatalogLookup, name="lookup")


def _function_call_output(namespace: str | None, arguments: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "status": "completed",
        "name": "lookup",
        "arguments": arguments,
    }
    if namespace is not None:
        output["namespace"] = namespace
    return output


def test_get_input_tool_by_name_resolves_namespaced_call() -> None:
    namespace_tool = _crm_namespace_tool()
    nested = namespace_tool["tools"][0]

    tool = get_input_tool_by_name(
        input_tools=[namespace_tool],
        name="lookup",
        namespace="crm",
    )

    assert tool is nested


def test_get_input_tool_by_name_namespaced_call_does_not_match_top_level_tool() -> None:
    tool = get_input_tool_by_name(
        input_tools=[_catalog_top_level_tool(), _crm_namespace_tool()],
        name="lookup",
        namespace="crm",
    )

    assert tool is not None
    assert isinstance(tool, openai.lib._tools.ResponsesPydanticFunctionTool)
    assert tool.model is CrmLookup


def test_get_input_tool_by_name_unknown_namespace_returns_none() -> None:
    tool = get_input_tool_by_name(
        input_tools=[_catalog_top_level_tool()],
        name="lookup",
        namespace="unknown",
    )

    assert tool is None


def test_get_input_tool_by_name_without_namespace_still_matches_top_level_tool() -> None:
    catalog = _catalog_top_level_tool()

    tool = get_input_tool_by_name(input_tools=[catalog], name="lookup")

    assert tool is catalog


def _mock_response_body(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": "gpt-4o-mini",
        "output": [output],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "reasoning": {"effort": None, "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": None,
        "user": None,
        "metadata": {},
    }


@pytest.mark.respx2(base_url=base_url)
def test_parse_resolves_namespaced_call_inside_namespace_tool(client: OpenAI, respx2_mock: MockRouter) -> None:
    respx2_mock.post("/responses").respond(
        json=_mock_response_body(_function_call_output("crm", json.dumps({"customer_id": "C-42"})))
    )

    response = client.responses.parse(
        model="gpt-4o-mini",
        input="Look up the customer",
        tools=[_crm_namespace_tool()],
    )

    call = response.output[0]
    assert isinstance(call, openai.types.responses.ParsedResponseFunctionToolCall)
    assert isinstance(call.parsed_arguments, CrmLookup)
    assert call.parsed_arguments.customer_id == "C-42"


@pytest.mark.respx2(base_url=base_url)
def test_parse_namespaced_call_does_not_match_same_named_top_level_tool(
    client: OpenAI, respx2_mock: MockRouter
) -> None:
    respx2_mock.post("/responses").respond(
        json=_mock_response_body(_function_call_output("crm", json.dumps({"customer_id": "C-42"})))
    )

    response = client.responses.parse(
        model="gpt-4o-mini",
        input="Look up the customer",
        tools=[_catalog_top_level_tool(), _crm_namespace_tool()],
    )

    call = response.output[0]
    assert isinstance(call, openai.types.responses.ParsedResponseFunctionToolCall)
    assert isinstance(call.parsed_arguments, CrmLookup)


@pytest.mark.respx2(base_url=base_url)
def test_parse_non_namespaced_call_still_matches_top_level_tool(client: OpenAI, respx2_mock: MockRouter) -> None:
    respx2_mock.post("/responses").respond(
        json=_mock_response_body(_function_call_output(None, json.dumps({"sku": 1})))
    )

    response = client.responses.parse(
        model="gpt-4o-mini",
        input="Look up the catalog entry",
        tools=[_catalog_top_level_tool()],
    )

    call = response.output[0]
    assert isinstance(call, openai.types.responses.ParsedResponseFunctionToolCall)
    assert isinstance(call.parsed_arguments, CatalogLookup)
    assert call.parsed_arguments.sku == 1
