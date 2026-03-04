from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from copy import deepcopy
from typing import Annotated

import httpx
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, model_validator

from app.skills import SkillStore

logger = logging.getLogger(__name__)

mcp = FastMCP("agent-tools")
_skill_store: SkillStore | None = None
_openapi_loaded = False
_openapi_spec_path = Path(__file__).with_name("openapi_fake_app_agent_api.json")


class GetSkillsInput(BaseModel):
    skill_id: str | None = Field(
        default=None,
        description="Single skill_id to load. Prefer skill_ids for loading multiple skills in one call.",
    )
    skill_ids: list[str] | None = Field(
        default=None,
        description="List of skill_ids to load in a single call. Prefer this to reduce tool round-trips.",
    )

    @model_validator(mode="after")
    def _validate_inputs(self) -> "GetSkillsInput":
        if self.skill_id is None and not self.skill_ids:
            raise ValueError("Provide skill_id or skill_ids")
        if self.skill_id is not None and self.skill_ids:
            raise ValueError("Provide either skill_id or skill_ids, not both")
        return self


class WebRequestInput(BaseModel):
    method: str = Field(default="GET", description="HTTP method")
    url: str = Field(..., description="Target URL")
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = Field(default=None)


class HouseSearchInput(BaseModel):
    district: str | None = Field(default=None, description="行政区")
    area: str | None = Field(default=None, description="商圈，逗号分隔")
    min_price: int | None = Field(default=None, description="最低月租金（元）")
    max_price: int | None = Field(default=None, description="最高月租金（元）")
    bedrooms: str | None = Field(default=None, description="卧室数，逗号分隔")
    rental_type: str | None = Field(default=None, description="整租 或 合租")
    decoration: str | None = Field(default=None, description="装修，如精装/简装")
    orientation: str | None = Field(default=None, description="朝向")
    elevator: str | None = Field(default=None, description="是否有电梯：true/false")
    min_area: int | None = Field(default=None, description="最小面积（平米）")
    max_area: int | None = Field(default=None, description="最大面积（平米）")
    property_type: str | None = Field(default=None, description="物业类型")
    subway_line: str | None = Field(default=None, description="地铁线路")
    max_subway_dist: int | None = Field(default=None, description="最大地铁距离（米）")
    subway_station: str | None = Field(default=None, description="地铁站名")
    utilities_type: str | None = Field(default=None, description="水电类型")
    available_from_before: str | None = Field(default=None, description="可入住日期上限，YYYY-MM-DD")
    commute_to_xierqi_max: int | None = Field(default=None, description="到西二旗通勤时间上限（分钟）")
    sort_by: str | None = Field(default=None, description="排序字段：price/area/subway")
    sort_order: str | None = Field(default=None, description="排序顺序：asc/desc")
    page: int | None = Field(default=1, description="页码")
    page_size: int = Field(default=5, description="每页数量")


class HouseNearbySearchInput(BaseModel):
    landmark_id: str = Field(..., description="地标ID")
    max_distance: int | None = Field(default=None, description="最大距离（米）")
    page: int | None = Field(default=1, description="页码")
    page_size: int = Field(default=5, description="每页数量")


class HouseCommunitySearchInput(BaseModel):
    community: str = Field(..., description="小区名")
    page: int | None = Field(default=1, description="页码")
    page_size: int = Field(default=5, description="每页数量")


class LandmarkSearchInput(BaseModel):
    name: str = Field(..., description="地标名称")


def set_skill_store(skill_store: SkillStore) -> None:
    global _skill_store
    _skill_store = skill_store


def _require_skill_store() -> SkillStore:
    if _skill_store is None:
        raise RuntimeError("Skill store is not initialized")
    return _skill_store


# @mcp.tool(
#     name="get_skills",
#     description="Return full SKILL.md content for one or more skill_ids. Prefer skill_ids to fetch multiple skills in one call.",
# )
# def get_skills_mcp(skill_id: str | None = None, skill_ids: list[str] | None = None) -> str:
#     return get_skills(skill_store=_require_skill_store(), skill_id=skill_id, skill_ids=skill_ids)
def get_skills(
    skill_store: SkillStore,
    skill_id: str | None = None,
    skill_ids: list[str] | None = None,
) -> str:
    requested_ids = [skill_id] if skill_id is not None else (skill_ids or [])
    logger.info("get_skills called | skill_ids=%s", requested_ids)

    found: list[dict[str, str]] = []
    unknown: list[str] = []
    for requested_id in requested_ids:
        skill = skill_store.get(requested_id)
        if not skill:
            unknown.append(requested_id)
            continue
        found.append({"skill_id": requested_id, "content": skill.body})

    payload: dict[str, object] = {"skills": found}
    if unknown:
        payload["errors"] = [{"skill_id": sid, "error": f"Unknown skill_id: {sid}"} for sid in unknown]
    return json.dumps(payload)


# @mcp.tool(name="web_request", description="Perform an HTTP request and return status, headers, and truncated body.")
# def web_request_mcp(
#     method: str = "GET",
#     url: str = "",
#     headers: dict[str, str] | None = None,
#     body: str | None = None,
# ) -> str:
#     return web_request(method=method, url=url, headers=headers, body=body)
def web_request(method: str, url: str, headers: dict[str, str] | None = None, body: str | None = None) -> str:
    logger.info("web_request called | method=%s | url=%s | headers=%s | body_preview=%s", method, url, headers or {}, (body or "")[:300])
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            response = client.request(method=method.upper(), url=url, headers=headers or {}, content=body)
        truncated = response.text[:4000]
        payload = json.dumps(
            {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": truncated,
            }
        )
        logger.info("web_request result | status=%s | body_preview=%s", response.status_code, truncated[:300])
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.exception("web_request failed")
        return json.dumps({"error": str(exc)})


def _extract_house_ids(payload: object) -> list[str]:
    if not isinstance(payload, (dict, list)):
        return []

    def _extract_items(node: object) -> list[dict[str, object]]:
        if isinstance(node, list):
            return [item for item in node if isinstance(item, dict)]
        if isinstance(node, dict):
            for key in ("houses", "items", "results"):
                value = node.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]

            data_value = node.get("data")
            if isinstance(data_value, (dict, list)):
                return _extract_items(data_value)

        return []

    items = _extract_items(payload)

    house_ids: list[str] = []
    for item in items:
        house_id = item.get("house_id")
        if isinstance(house_id, str):
            house_ids.append(house_id)

    return house_ids


def _append_house_platform(
    houses: list[dict[str, object]],
    house_platforms: dict[str, list[str]],
    house_id: str,
    platform: str,
) -> None:
    platforms = house_platforms.get(house_id)
    if platforms is None:
        house_platforms[house_id] = [platform]
        houses.append({"houseid": house_id, "platforms": house_platforms[house_id]})
        return
    if platform not in platforms:
        platforms.append(platform)


@mcp.tool(
    name="search_landmarks",
    description="根据地标名称搜索地标并返回首个匹配地标ID。输入name，输出id。",
)
def search_landmarks(name: Annotated[str, Field(description="地标名称")]) -> str:
    params = LandmarkSearchInput(name=name).model_dump()
    with _openapi_spec_path.open("r", encoding="utf-8") as fp:
        openapi_spec = json.load(fp)
    server = openapi_spec.get("servers", [{}])[0].get("url", "")

    try:
        with httpx.Client(base_url=server, timeout=20, trust_env=False) as client:
            response = client.get("/api/landmarks/search", params={"q": params["name"]})
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_landmarks failed | name=%s", name)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    items: list[dict[str, object]] = []
    if isinstance(payload, list):
        items = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        for key in ("items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                items = [item for item in value if isinstance(item, dict)]
                break

    if not items:
        return json.dumps({"name": name, "id": None}, ensure_ascii=False)

    first = items[0]
    landmark_id = first.get("id") if isinstance(first.get("id"), str) else None
    landmark_name = first.get("name") if isinstance(first.get("name"), str) else name
    return json.dumps({"name": landmark_name, "id": landmark_id}, ensure_ascii=False)


@mcp.tool(
    name="search_house",
    description="根据筛选条件搜索可租房源，默认获取三大平台（链家/安居客/58同城）的所有符合条件的房源。"
                "填写条件时请仅包含**必须的要求**，“如果可以” “有的话更好” “最好有” 等可选条件不能包括。",
)
def search_house(
    district: Annotated[str | None, Field(description="仅支持以下北京行政区：朝阳、西城、海淀、东城、丰台、昌平、房山、通州、大兴、顺义")] = None,
    area: Annotated[str | None, Field(description="商圈，多个可使用逗号分隔")] = None,
    min_price: Annotated[int | None, Field(description="最低月租金（元）")] = None,
    max_price: Annotated[int | None, Field(description="最高月租金（元）")] = None,
    bedrooms: Annotated[str | None, Field(description="卧室数，多个可使用逗号分隔。如两到三居室：2,3")] = None,
    rental_type: Annotated[str | None, Field(description="支持：整租、合租")] = None,
    decoration: Annotated[str | None, Field(description="装修类型：精装、简装")] = None,
    orientation: Annotated[str | None, Field(description="朝向：朝南、南北、朝北、朝西、东西、朝东、西北")] = None,
    elevator: Annotated[str | None, Field(description="是否有电梯：true/false")] = None,
    min_area: Annotated[int | None, Field(description="最小面积（平米）")] = None,
    max_area: Annotated[int | None, Field(description="最大面积（平米）")] = None,
    property_type: Annotated[str | None, Field(description="物业类型：住宅、公寓")] = None,
    subway_line: Annotated[str | None, Field(description="地铁线路")] = None,
    max_subway_dist: Annotated[int | None, Field(description="最大地铁距离（米）")] = None,
    subway_station: Annotated[str | None, Field(description="地铁站名，使用前必须通过地标搜索获取精准名称")] = None,
    utilities_type: Annotated[str | None, Field(description="水电类型：民水民电、商水商电")] = None,
    available_from_before: Annotated[str | None, Field(description="可入住日期上限，YYYY-MM-DD")] = None,
    commute_to_xierqi_max: Annotated[int | None, Field(description="到西二旗通勤时间上限（分钟）")] = None,
    sort_by: Annotated[str | None, Field(description="排序字段：price/area/subway")] = None,
    sort_order: Annotated[str | None, Field(description="排序顺序：asc/desc")] = None,
    page: Annotated[int | None, Field(description="页码")] = 1,
    page_size: Annotated[int, Field(description="每页数量")] = 5,
    # detailed: Annotated[bool, Field(description="是否返回HTTP接口完整结果；true时不裁剪")] = False,
    final_answer: Annotated[bool, Field(description="当该次调用是用户请求的最终结果时，请填写true，该结果会直接提供给用户。")] = False,
) -> str:
    params = HouseSearchInput(
        district=district,
        area=area,
        min_price=min_price,
        max_price=max_price,
        bedrooms=bedrooms,
        rental_type=rental_type,
        decoration=decoration,
        orientation=orientation,
        elevator=elevator,
        min_area=min_area,
        max_area=max_area,
        property_type=property_type,
        subway_line=subway_line,
        max_subway_dist=max_subway_dist,
        subway_station=subway_station,
        utilities_type=utilities_type,
        available_from_before=available_from_before,
        commute_to_xierqi_max=commute_to_xierqi_max,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        page_size=page_size,
    ).model_dump(exclude_none=True)
    detailed = False
    with _openapi_spec_path.open("r", encoding="utf-8") as fp:
        openapi_spec = json.load(fp)
    server = openapi_spec.get("servers", [{}])[0].get("url", "")

    platforms = ["链家", "安居客", "58同城"]
    houses: list[dict[str, object]] = []
    house_platforms: dict[str, list[str]] = {}
    raw_results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    with httpx.Client(base_url=server, timeout=20, trust_env=False, headers={"X-User-ID": "d00640449"}) as client:
        for platform in platforms:
            request_params = {**params, "listing_platform": platform}
            try:
                response = client.get("/api/houses/by_platform", params=request_params)
                response.raise_for_status()
                payload = response.json()
                if detailed:
                    raw_results.append({"platform": platform, "result": payload})
                house_ids = _extract_house_ids(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("house_search failed for platform=%s", platform)
                errors.append({"platform": platform, "error": str(exc)})
                house_ids = []

            if not detailed:
                for house_id in house_ids:
                    _append_house_platform(houses, house_platforms, house_id, platform)

    result: dict[str, object] = {"raw_results": raw_results} if detailed else {"houses": houses}
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="get_houses_near_landmark",
    description="以地标为圆心，查询在指定距离内的可租房源，返回带直线距离、步行距离、步行时间。必须先使用search_landmarks获取精准地标名称/ID后，才能调用该接口。"
                "默认会查询三大平台（链家/安居客/58同城）的符合条件的房源。",
)
def get_houses_near_landmark(
    landmark_id: Annotated[str, Field(description="地标ID")],
    max_distance: Annotated[int | None, Field(description="最大距离（米）")] = None,
    page: Annotated[int | None, Field(description="页码")] = 1,
    page_size: Annotated[int, Field(description="每页数量")] = 5,
    # detailed: Annotated[bool, Field(description="是否返回HTTP接口完整结果；true时不裁剪")] = False,
    final_answer: Annotated[bool, Field(description="当该次调用是用户请求的最终结果时，请填写true，该结果会直接提供给用户。")] = False,
) -> str:
    params = HouseNearbySearchInput(
        landmark_id=landmark_id,
        max_distance=max_distance,
        page=page,
        page_size=page_size,
    ).model_dump(exclude_none=True)
    detailed = False
    with _openapi_spec_path.open("r", encoding="utf-8") as fp:
        openapi_spec = json.load(fp)
    server = openapi_spec.get("servers", [{}])[0].get("url", "")

    platforms = ["链家", "安居客", "58同城"]
    houses: list[dict[str, object]] = []
    house_platforms: dict[str, list[str]] = {}
    raw_results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    with httpx.Client(base_url=server, timeout=20, trust_env=False, headers={"X-User-ID": "d00640449"}) as client:
        for platform in platforms:
            request_params = {**params, "listing_platform": platform}
            try:
                response = client.get("/api/houses/nearby", params=request_params)
                response.raise_for_status()
                payload = response.json()
                if detailed:
                    raw_results.append({"platform": platform, "result": payload})
                house_ids = _extract_house_ids(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("get_houses_list_nearby failed for platform=%s", platform)
                errors.append({"platform": platform, "error": str(exc)})
                house_ids = []

            if not detailed:
                for house_id in house_ids:
                    _append_house_platform(houses, house_platforms, house_id, platform)

    result: dict[str, object] = {"raw_results": raw_results} if detailed else {"houses": houses}
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="get_houses_by_community",
    description="按小区名查询可租房源，默认会查询三大平台（链家/安居客/58同城）的所有符合条件的房源。",
)
def get_houses_by_community(
    community: Annotated[str, Field(description="小区名")],
    page: Annotated[int | None, Field(description="页码")] = 1,
    page_size: Annotated[int, Field(description="每页数量")] = 5,
    # detailed: Annotated[bool, Field(description="是否返回HTTP接口完整结果；true时不裁剪")] = False,
    final_answer: Annotated[
        bool,
        Field(description="当该次调用是用户请求的最终结果时，请填写true，该结果会直接提供给用户。"),
    ] = False,
) -> str:
    params = HouseCommunitySearchInput(
        community=community,
        page=page,
        page_size=page_size,
    ).model_dump(exclude_none=True)
    detailed = False
    with _openapi_spec_path.open("r", encoding="utf-8") as fp:
        openapi_spec = json.load(fp)
    server = openapi_spec.get("servers", [{}])[0].get("url", "")

    platforms = ["链家", "安居客", "58同城"]
    houses: list[dict[str, object]] = []
    house_platforms: dict[str, list[str]] = {}
    raw_results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    with httpx.Client(base_url=server, timeout=20, trust_env=False, headers={"X-User-ID": "d00640449"}) as client:
        for platform in platforms:
            request_params = {**params, "listing_platform": platform}
            try:
                response = client.get("/api/houses/by_community", params=request_params)
                response.raise_for_status()
                payload = response.json()
                if detailed:
                    raw_results.append({"platform": platform, "result": payload})
                house_ids = _extract_house_ids(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("get_houses_list_by_community failed for platform=%s", platform)
                errors.append({"platform": platform, "error": str(exc)})
                house_ids = []

            if not detailed:
                for house_id in house_ids:
                    _append_house_platform(houses, house_platforms, house_id, platform)

    result: dict[str, object] = {"raw_results": raw_results} if detailed else {"houses": houses}
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


def _normalize_openapi_spec(spec: dict) -> dict:
    normalized = deepcopy(spec)
    normalized.get("paths", {}).pop("/api/landmarks/search", None)
    for path_item in normalized.get("paths", {}).values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            operation.setdefault(
                "responses",
                {"200": {"description": "OK", "content": {"application/json": {"schema": {"type": "object"}}}}},
            )
    return normalized



class AgentTools:
    def __init__(self, skill_store: SkillStore) -> None:
        self.skill_store = skill_store
        set_skill_store(skill_store)
        self._ensure_openapi_tools_loaded()

    def _ensure_openapi_tools_loaded(self) -> None:
        global _openapi_loaded
        if _openapi_loaded:
            return

        async def _load() -> None:
            with _openapi_spec_path.open("r", encoding="utf-8") as fp:
                openapi_spec = _normalize_openapi_spec(json.load(fp))

            server = openapi_spec.get("servers", [{}])[0].get("url", "")
            # OpenAPI-generated tools are async and await the configured HTTP
            # client request call. Passing httpx.Client (sync) causes runtime
            # failures such as: "object Response can't be used in 'await'
            # expression" for imported operations.
            child_server = FastMCP.from_openapi(
                openapi_spec,
                httpx.AsyncClient(base_url=server, timeout=20, trust_env=False, headers={"X-User-ID": "d00640449"}),
            )
            mcp.mount(server=child_server, namespace="")

        try:
            asyncio.run(_load())
            _openapi_loaded = True
            logger.info("Loaded OpenAPI tools from %s", _openapi_spec_path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load OpenAPI tools from %s", _openapi_spec_path)

    async def _mcp_tools_async(self, allowed_tools: set[str] | None = None) -> list[dict[str, object]]:
        if allowed_tools is not None and not allowed_tools:
            return []
        tools = await mcp.list_tools()
        allowed = {name.lower() for name in allowed_tools} if allowed_tools is not None else None
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
            if allowed is None or tool.name.lower() in allowed
        ]
        logger.info("MCP tools | %s", tools)
        return tools

    async def langchain_tools(self, allowed_tools: set[str] | None = None) -> list[dict[str, object]]:
        return await self._mcp_tools_async(allowed_tools=allowed_tools)

    async def _dispatch_tool_async(self, name: str, args: dict) -> str:
        async with Client(mcp) as client:
            try:
                result = await client.call_tool(name, args)
            except ToolError as err:
                return json.dumps({"error": str(err)})
        if result.is_error:
            return json.dumps({"error": result.data})
        return str(result.data)

    async def dispatch_tool(self, name: str, args: dict) -> str:
        logger.info("dispatch_tool called | name=%s | args=%s", name, json.dumps(args, ensure_ascii=False))
        try:
            result = await self._dispatch_tool_async(name, args)
        except Exception:  # noqa: BLE001
            logger.exception("dispatch_tool failed")
            return json.dumps({"error": f"Unknown tool {name}"})
        # logger.info("dispatch_tool finished | name=%s | result_preview=%s", name, result[:500])
        return result
