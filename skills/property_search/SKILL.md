---
name: property_search
description: 按条件搜索/查看/对比可租赁的房源，只要上下文出现与房屋租赁、查询有关的，使用该技能。
allowed-tools: search_landmarks,get_landmark_by_id,get_landmark_stats,search_house,get_houses_by_community,get_houses_near_landmark,get_nearby_landmarks,get_house_by_id,get_house_listings,get_house_stats
---

# Prompt
- 目标：快速找到符合条件的房源，禁止编造信息。
- 先提取条件：预算、区域/商圈、户型、整租/合租、地铁/通勤、入住时间。
- 条件清晰就直接调用搜索工具；不清晰只问 1 个关键追问。
- 如需地标/地铁附近搜索，先用地标工具拿到 `landmark_id`。
- 同一房源需要细节或跨平台对比时，调用 `get_house_by_id` 与 `get_house_listings`。
- 返回简短结论，最多推荐 5 套，附 house_id。

# Tool mapping
- 常规筛选：`get_houses_by_platform`
- 按小区：`get_houses_by_community`
- 地标附近：`get_houses_nearby`
- 房源详情：`get_house_by_id`
- 跨平台挂牌：`get_house_listings`
- 地标解析：`get_landmarks` / `get_landmark_by_name` / `search_landmarks` / `get_landmark_by_id`
- 统计参考：`get_landmark_stats` / `get_house_stats` / `get_nearby_landmarks`
