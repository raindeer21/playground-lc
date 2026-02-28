from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from copy import deepcopy

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


class ProvidePropertyResultListInput(BaseModel):
    message: str = Field(default="为您找到以下符合条件的房源：", description="User-facing message")
    houses: list[str] = Field(default_factory=list, description="Matched house_id list")


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


def set_skill_store(skill_store: SkillStore) -> None:
    global _skill_store
    _skill_store = skill_store


def _require_skill_store() -> SkillStore:
    if _skill_store is None:
        raise RuntimeError("Skill store is not initialized")
    return _skill_store


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


# @mcp.tool(
#     name="get_skills",
#     description="Return full SKILL.md content for one or more skill_ids. Prefer skill_ids to fetch multiple skills in one call.",
# )
# def get_skills_mcp(skill_id: str | None = None, skill_ids: list[str] | None = None) -> str:
#     return get_skills(skill_store=_require_skill_store(), skill_id=skill_id, skill_ids=skill_ids)


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


# @mcp.tool(name="web_request", description="Perform an HTTP request and return status, headers, and truncated body.")
# def web_request_mcp(
#     method: str = "GET",
#     url: str = "",
#     headers: dict[str, str] | None = None,
#     body: str | None = None,
# ) -> str:
#     return web_request(method=method, url=url, headers=headers, body=body)


@mcp.tool(
    name="current_properties",
    description=(
        "提供与本次对话相关的房源信息。"
            "触发条件：用户提出找房/搜索/推荐/筛选房源（按条件过滤）、查看房源详情、对比各平台挂牌信息，"
            "或执行租房、退租/解除租约、下架等操作；"
            "也适用于对已讨论过的房源进行总结或再次引用时。"
    ),
)
def current_properties(message: str = "为您找到以下符合条件的房源：", houses: list[str] | None = None) -> str:
    payload = ProvidePropertyResultListInput(message=message, houses=houses or [])
    return payload.model_dump_json(ensure_ascii=False)


def _extract_house_ids(payload: object) -> list[str]:
    if not isinstance(payload, (dict, list)):
        return []

    items: list[dict[str, object]] = []
    if isinstance(payload, list):
        items = [item for item in payload if isinstance(item, dict)]
    else:
        for key in ("houses", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                items = [item for item in value if isinstance(item, dict)]
                break

    house_ids: list[str] = []
    for item in items:
        house_id = item.get("house_id")
        if isinstance(house_id, str):
            house_ids.append(house_id)

    return house_ids


@mcp.tool(
    name="get_houses_by_platform_simple",
    description="触发器：不需要具体信息，仅搜索房源列表时优先使用该工具。"
                "仅当需要房屋的具体信息时使用get_houses_by_platform。"
                "根据筛选条件分别获取三大平台（链家/安居客/58同城）的房源，并返回 {houseid, platform} 列表。",
)
def get_houses_by_platform_simple(
    district: str | None = None,
    area: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    bedrooms: str | None = None,
    rental_type: str | None = None,
    decoration: str | None = None,
    orientation: str | None = None,
    elevator: str | None = None,
    min_area: int | None = None,
    max_area: int | None = None,
    property_type: str | None = None,
    subway_line: str | None = None,
    max_subway_dist: int | None = None,
    subway_station: str | None = None,
    utilities_type: str | None = None,
    available_from_before: str | None = None,
    commute_to_xierqi_max: int | None = None,
    sort_by: str | None = None,
    sort_order: str | None = None,
    page: int | None = 1,
    page_size: int = 5,
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

    with _openapi_spec_path.open("r", encoding="utf-8") as fp:
        openapi_spec = json.load(fp)
    server = openapi_spec.get("servers", [{}])[0].get("url", "")

    platforms = ["链家", "安居客", "58同城"]
    houses: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    with httpx.Client(base_url=server, timeout=20, trust_env=False, headers={"X-User-ID": "d00640449"}) as client:
        for platform in platforms:
            request_params = {**params, "listing_platform": platform}
            try:
                response = client.get("/api/houses/by_platform", params=request_params)
                response.raise_for_status()
                payload = response.json()
                house_ids = _extract_house_ids(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("house_search failed for platform=%s", platform)
                errors.append({"platform": platform, "error": str(exc)})
                house_ids = []

            for house_id in house_ids:
                houses.append({"houseid": house_id, "platform": platform})

    result: dict[str, object] = {
        "houses": houses,
    }
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="get_houses_nearby_simple",
    description="触发器：不需要具体信息，仅搜索房源列表时优先使用该工具。"
                "仅当需要房屋的具体信息时使用 get_houses_nearby。"
                "以地标为圆心查附近房源，分别获取三大平台（链家/安居客/58同城）的房源，并返回 {houseid, platform} 列表。"
                "必须先使用search_landmarks获取精准地标名称/ID后，才能调用该接口。以地标为圆心，查询在指定距离内的可租房源，返回带直线距离、步行距离、步行时间。",
)
def get_houses_nearby_simple(
    landmark_id: str,
    max_distance: int | None = None,
    page: int | None = 1,
    page_size: int = 5,
) -> str:
    params = HouseNearbySearchInput(
        landmark_id=landmark_id,
        max_distance=max_distance,
        page=page,
        page_size=page_size,
    ).model_dump(exclude_none=True)

    with _openapi_spec_path.open("r", encoding="utf-8") as fp:
        openapi_spec = json.load(fp)
    server = openapi_spec.get("servers", [{}])[0].get("url", "")

    platforms = ["链家", "安居客", "58同城"]
    houses: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    with httpx.Client(base_url=server, timeout=20, trust_env=False, headers={"X-User-ID": "d00640449"}) as client:
        for platform in platforms:
            request_params = {**params, "listing_platform": platform}
            try:
                response = client.get("/api/houses/nearby", params=request_params)
                response.raise_for_status()
                payload = response.json()
                house_ids = _extract_house_ids(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("get_houses_list_nearby failed for platform=%s", platform)
                errors.append({"platform": platform, "error": str(exc)})
                house_ids = []

            for house_id in house_ids:
                houses.append({"houseid": house_id, "platform": platform})

    result: dict[str, object] = {
        "houses": houses,
    }
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="get_houses_by_community",
    description="根据小区名分别调用三大平台（链家/安居客/58同城）的 get_houses_by_community 接口，并返回 {houseid, platform} 列表。",
)
def get_houses_list_by_community(
    community: str,
    page: int | None = 1,
    page_size: int = 5,
) -> str:
    params = HouseCommunitySearchInput(
        community=community,
        page=page,
        page_size=page_size,
    ).model_dump(exclude_none=True)

    with _openapi_spec_path.open("r", encoding="utf-8") as fp:
        openapi_spec = json.load(fp)
    server = openapi_spec.get("servers", [{}])[0].get("url", "")

    platforms = ["链家", "安居客", "58同城"]
    houses: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    with httpx.Client(base_url=server, timeout=20, trust_env=False, headers={"X-User-ID": "d00640449"}) as client:
        for platform in platforms:
            request_params = {**params, "listing_platform": platform}
            try:
                response = client.get("/api/houses/by_community", params=request_params)
                response.raise_for_status()
                payload = response.json()
                house_ids = _extract_house_ids(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("get_houses_list_by_community failed for platform=%s", platform)
                errors.append({"platform": platform, "error": str(exc)})
                house_ids = []

            for house_id in house_ids:
                houses.append({"houseid": house_id, "platform": platform})

    result: dict[str, object] = {
        "houses": houses,
    }
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


def _normalize_openapi_spec(spec: dict) -> dict:
    normalized = deepcopy(spec)
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

    async def _mcp_tools_async(self) -> list[dict[str, object]]:
        tools = await mcp.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]

    async def langchain_tools(self) -> list[dict[str, object]]:
        return await self._mcp_tools_async()

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
