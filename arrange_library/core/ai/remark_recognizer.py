"""
备注识别服务 - 使用Azure OpenAI进行上机备注结构化识别（v2）
创建时间：2025-11-17 00:00:00
更新时间：2026-01-16 14:49:06
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from liblane_paths import setup_liblane_paths

setup_liblane_paths()

from core.ai.azure_openai_client import AzureOpenAIClient
from models.library_info import EnhancedLibraryInfo
from models.remark_recognition_v2 import CommandItem, RemarkRecognitionResultV2

SYSTEM_START_MARKER: str = "=====SYSTEM_PROMPT_START====="
SYSTEM_END_MARKER: str = "=====SYSTEM_PROMPT_END====="
USER_START_MARKER: str = "=====USER_PROMPT_TEMPLATE_START====="
USER_END_MARKER: str = "=====USER_PROMPT_TEMPLATE_END====="
TYPE_START_MARKER: str = "=====TYPE_PROMPT_START====="
TYPE_END_MARKER: str = "=====TYPE_PROMPT_END====="

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORE_PROMPT_DOC: str = str(PROJECT_ROOT / "prompts/remark_recognition/core_prompt_doc.md")
DEFAULT_ROUTER_PROMPT_DOC: str = str(PROJECT_ROOT / "prompts/remark_recognition/router_prompt_doc.md")
DEFAULT_TYPE_REGISTRY: str = str(PROJECT_ROOT / "prompts/remark_recognition/type_registry.json")

ALLOWED_IS_NEED = {"需识别", "人工识别", "忽略"}


@dataclass
class RemarkTypeDef:
    """业务语义类型定义（用于规则路由与拼接类型片段）"""

    type_id: str
    name: str
    description: str
    prompt_doc_path: Path
    priority: int
    regex_patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    route_hints: List[str] = field(default_factory=list)
    compiled_regex: List[re.Pattern] = field(default_factory=list)


class RemarkRecognizer:
    """备注识别服务 - v2（is_need/explain/commands）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        model: str = "gpt-4o",
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        confidence_threshold: float = 0.3,
        concurrent_requests: int = 5,
        core_prompt_doc: str = DEFAULT_CORE_PROMPT_DOC,
        router_prompt_doc: str = DEFAULT_ROUTER_PROMPT_DOC,
        type_registry: str = DEFAULT_TYPE_REGISTRY,
    ) -> None:
        self.client = AzureOpenAIClient(
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            model=model,
        )
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.confidence_threshold = confidence_threshold
        self.concurrent_requests = concurrent_requests
        self._cache: Dict[str, Tuple[RemarkRecognitionResultV2, float]] = {}

        self.core_system_prompt, self.user_prompt_template = self._load_prompt_doc(
            core_prompt_doc
        )
        self.router_system_prompt, self.router_user_prompt_template = self._load_prompt_doc(
            router_prompt_doc
        )
        self.type_defs = self._load_type_registry(Path(type_registry))
        self.type_catalog_json = self._build_type_catalog_json(self.type_defs)
        self._type_addendum_cache: Dict[str, str] = {}

        logger.info(
            "备注识别器(v2)初始化完成 - 置信度阈值: {}".format(confidence_threshold)
        )

    def _get_cache_key(self, remark_text: str) -> str:
        text_hash = hashlib.md5(remark_text.encode("utf-8")).hexdigest()
        return f"remark_recognition_v2:{text_hash}"

    def _get_cached_result(self, cache_key: str) -> Optional[RemarkRecognitionResultV2]:
        if not self.cache_enabled:
            return None
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"使用缓存结果: {cache_key}")
                return result
            del self._cache[cache_key]
        return None

    def _set_cached_result(self, cache_key: str, result: RemarkRecognitionResultV2) -> None:
        if self.cache_enabled:
            self._cache[cache_key] = (result, time.time())

    @staticmethod
    def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"提示词文档缺少标记: {start_marker} / {end_marker}")
        start_idx += len(start_marker)
        return text[start_idx:end_idx].strip()

    def _load_prompt_doc(self, doc_path: str) -> Tuple[str, str]:
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"未找到提示词文档: {path}")
        content = path.read_text(encoding="utf-8")
        system_prompt = self._extract_block(content, SYSTEM_START_MARKER, SYSTEM_END_MARKER)
        user_prompt = self._extract_block(content, USER_START_MARKER, USER_END_MARKER)
        if "{seqnotes_value}" not in user_prompt:
            raise ValueError("user_prompt_template 缺少 {seqnotes_value} 占位符")
        return system_prompt, user_prompt

    def _load_type_registry(self, registry_path: Path) -> List[RemarkTypeDef]:
        if not registry_path.exists():
            raise FileNotFoundError(f"未找到类型注册表: {registry_path}")
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        raw_types = data.get("types")
        if not isinstance(raw_types, list) or not raw_types:
            raise ValueError("type_registry.json 缺少有效的 types 列表")

        type_defs: List[RemarkTypeDef] = []
        for t in raw_types:
            if not isinstance(t, dict):
                continue
            type_id = str(t.get("type_id", "")).strip()
            if not type_id:
                continue
            prompt_path = Path(str(t.get("prompt_doc_path", "")).strip())
            if not prompt_path.is_absolute():
                prompt_path = (PROJECT_ROOT / prompt_path).resolve()

            rule_route = t.get("rule_route") if isinstance(t.get("rule_route"), dict) else {}
            regex_patterns = rule_route.get("regex_patterns") if isinstance(
                rule_route.get("regex_patterns"), list
            ) else []
            keywords = rule_route.get("keywords") if isinstance(
                rule_route.get("keywords"), list
            ) else []

            compiled: List[re.Pattern] = []
            for p in regex_patterns:
                try:
                    compiled.append(re.compile(str(p), re.IGNORECASE))
                except re.error:
                    logger.warning(f"路由正则不可用: {p}")
                    continue

            type_defs.append(
                RemarkTypeDef(
                    type_id=type_id,
                    name=str(t.get("name", "")).strip(),
                    description=str(t.get("description", "")).strip(),
                    prompt_doc_path=prompt_path,
                    priority=int(t.get("priority", 9999)),
                    regex_patterns=[str(x) for x in regex_patterns],
                    keywords=[str(x) for x in keywords],
                    route_hints=[str(x) for x in (t.get("route_hints") or [])]
                    if isinstance(t.get("route_hints"), list)
                    else [],
                    compiled_regex=compiled,
                )
            )

        type_defs.sort(key=lambda x: x.priority)
        return type_defs

    @staticmethod
    def _build_type_catalog_json(type_defs: List[RemarkTypeDef]) -> str:
        catalog = []
        for t in type_defs:
            catalog.append(
                {
                    "type_id": t.type_id,
                    "name": t.name,
                    "description": t.description,
                    "route_hints": t.route_hints,
                }
            )
        return json.dumps(catalog, ensure_ascii=False)

    @staticmethod
    def _build_extraction_system_prompt(
        core_system_prompt: str, type_addendum: str, type_id: str
    ) -> str:
        return f"{core_system_prompt}\n\n{type_addendum}\n\n【已路由类型】{type_id}"

    def _build_user_prompt(self, seqnotes_value: str, explain: str = "") -> str:
        return self.user_prompt_template.format(
            seqnotes_value=seqnotes_value, explain=explain or ""
        )

    def _load_type_addendum(self, type_doc_path: Path) -> str:
        cache_key = str(type_doc_path)
        if cache_key in self._type_addendum_cache:
            return self._type_addendum_cache[cache_key]
        if not type_doc_path.exists():
            raise FileNotFoundError(f"未找到类型片段: {type_doc_path}")
        content = type_doc_path.read_text(encoding="utf-8")
        addendum = self._extract_block(content, TYPE_START_MARKER, TYPE_END_MARKER)
        self._type_addendum_cache[cache_key] = addendum
        return addendum

    def _rule_route_type_id(self, seqnotes_value: str) -> Optional[str]:
        text = (seqnotes_value or "").strip()
        if not text:
            return None
        lowered = text.lower()
        for t in self.type_defs:
            for cre in t.compiled_regex:
                if cre.search(text):
                    return t.type_id
            for kw in t.keywords:
                if kw and kw.lower() in lowered:
                    return t.type_id
        return None

    @staticmethod
    def _parse_json_maybe(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
            return {"error": "返回内容不是JSON对象(dict)"}
        except Exception as exc:
            return {"error": f"JSON解析失败: {type(exc).__name__}: {exc}"}

    def _build_result_from_ai(
        self,
        library_id: str,
        original_text: str,
        ai_result: Dict[str, Any],
        route_type_id: Optional[str],
        route_source: Optional[str],
    ) -> RemarkRecognitionResultV2:
        is_need = str(ai_result.get("is_need", "")).strip()
        explain = str(ai_result.get("explain", "")).strip()
        confidence = ai_result.get("confidence", 0.0)
        commands_raw = ai_result.get("commands", [])

        if is_need not in ALLOWED_IS_NEED:
            return RemarkRecognitionResultV2.create_unrecognized(
                library_id=library_id,
                original_text=original_text,
                reason=f"非法is_need值: {is_need}",
            )

        if is_need in {"人工识别", "忽略"}:
            commands_raw = []

        if not isinstance(commands_raw, list):
            return RemarkRecognitionResultV2.create_unrecognized(
                library_id=library_id,
                original_text=original_text,
                reason="commands不是列表",
            )

        commands: List[CommandItem] = []
        for item in commands_raw:
            if not isinstance(item, dict):
                return RemarkRecognitionResultV2.create_unrecognized(
                    library_id=library_id,
                    original_text=original_text,
                    reason="commands元素不是对象",
                )
            cmd_type = str(item.get("type", "")).strip()
            if not cmd_type:
                return RemarkRecognitionResultV2.create_unrecognized(
                    library_id=library_id,
                    original_text=original_text,
                    reason="commands缺少type字段",
                )
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            commands.append(CommandItem(type=cmd_type, params=params))

        if is_need == "需识别" and confidence < self.confidence_threshold:
            return RemarkRecognitionResultV2.create_unrecognized(
                library_id=library_id,
                original_text=original_text,
                reason=f"置信度低于阈值: {confidence}",
            )

        return RemarkRecognitionResultV2(
            library_id=library_id,
            original_text=original_text,
            is_need=is_need,
            explain=explain,
            commands=commands,
            confidence=float(confidence or 0.0),
            route_type_id=route_type_id,
            route_source=route_source,
            error_message=None,
            is_recognized=True,
            raw_result=ai_result,
        )

    async def _call_router_model(
        self, seqnotes_value: str, explain: str = ""
    ) -> Dict[str, Any]:
        user_prompt = self.router_user_prompt_template.format(
            type_catalog_json=self.type_catalog_json,
            seqnotes_value=seqnotes_value,
            explain=explain or "",
        )
        messages = [
            {"role": "system", "content": self.router_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await self.client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        response_text = response["choices"][0]["message"]["content"]
        return self._parse_json_maybe(response_text)

    async def recognize_single_remark(
        self,
        library_id: str,
        remark_text: str,
        library_info: Optional[Dict[str, Any]] = None,
    ) -> RemarkRecognitionResultV2:
        start_time = time.time()
        cleaned_text = remark_text.strip() if remark_text else ""

        if not cleaned_text:
            return RemarkRecognitionResultV2.create_ignored(
                library_id=library_id,
                original_text=remark_text or "",
                reason="备注为空",
            )

        cache_key = self._get_cache_key(cleaned_text)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        route_source = "rule"
        route_type_id = self._rule_route_type_id(cleaned_text)

        try:
            if not route_type_id:
                route_source = "router"
                router_result = await self._call_router_model(cleaned_text, "")
                if "error" in router_result:
                    return RemarkRecognitionResultV2.create_unrecognized(
                        library_id=library_id,
                        original_text=cleaned_text,
                        reason=router_result.get("error", "router失败"),
                    )

                router_is_need = str(router_result.get("is_need", "")).strip()
                router_type_id = router_result.get("type_id")
                router_explain = str(router_result.get("explain", "")).strip()
                router_confidence = float(router_result.get("confidence", 0.0) or 0.0)

                if router_is_need not in ALLOWED_IS_NEED:
                    return RemarkRecognitionResultV2.create_unrecognized(
                        library_id=library_id,
                        original_text=cleaned_text,
                        reason=f"router返回非法is_need: {router_is_need}",
                    )

                if router_is_need in {"人工识别", "忽略"}:
                    result = RemarkRecognitionResultV2(
                        library_id=library_id,
                        original_text=cleaned_text,
                        is_need=router_is_need,
                        explain=router_explain or router_is_need,
                        commands=[],
                        confidence=router_confidence,
                        route_type_id=None,
                        route_source="router",
                        error_message=None,
                        is_recognized=True,
                        raw_result=router_result,
                    )
                    self._set_cached_result(cache_key, result)
                    return result

                if not router_type_id:
                    return RemarkRecognitionResultV2.create_unrecognized(
                        library_id=library_id,
                        original_text=cleaned_text,
                        reason="router未返回type_id",
                    )

                route_type_id = str(router_type_id).strip()

            type_def_map = {t.type_id: t for t in self.type_defs}
            if route_type_id not in type_def_map:
                return RemarkRecognitionResultV2.create_unrecognized(
                    library_id=library_id,
                    original_text=cleaned_text,
                    reason=f"未知type_id: {route_type_id}",
                )

            addendum = self._load_type_addendum(type_def_map[route_type_id].prompt_doc_path)
            system_prompt = self._build_extraction_system_prompt(
                self.core_system_prompt, addendum, route_type_id
            )
            user_prompt = self._build_user_prompt(cleaned_text, "")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            response_text = response["choices"][0]["message"]["content"]
            ai_result = self._parse_json_maybe(response_text)
            if "usage" in response:
                ai_result["tokens_used"] = response["usage"].get("total_tokens", 0)

            result = self._build_result_from_ai(
                library_id=library_id,
                original_text=cleaned_text,
                ai_result=ai_result,
                route_type_id=route_type_id,
                route_source=route_source,
            )
            self._set_cached_result(cache_key, result)
            return result
        except Exception as exc:
            processing_time = time.time() - start_time
            logger.error(f"识别备注失败 (文库: {library_id}): {exc}")
            return RemarkRecognitionResultV2.create_unrecognized(
                library_id=library_id,
                original_text=cleaned_text,
                reason=f"未识别退回: API调用失败 - {type(exc).__name__}",
            )

    async def recognize_remarks_batch(
        self,
        remarks: Dict[str, str],
        libraries: Optional[List[EnhancedLibraryInfo]] = None,
    ) -> Dict[str, RemarkRecognitionResultV2]:
        logger.info(f"开始批量识别备注(v2) - 数量: {len(remarks)}")

        results: Dict[str, RemarkRecognitionResultV2] = {}
        semaphore = asyncio.Semaphore(self.concurrent_requests)

        async def recognize_with_semaphore(lib_id: str, remark_text: str):
            async with semaphore:
                return await self.recognize_single_remark(lib_id, remark_text)

        tasks = [
            recognize_with_semaphore(lib_id, remark_text)
            for lib_id, remark_text in remarks.items()
        ]
        recognition_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (lib_id, remark_text) in enumerate(remarks.items()):
            result = recognition_results[i]
            if isinstance(result, Exception):
                logger.error(f"识别任务失败 (文库: {lib_id}): {result}")
                results[lib_id] = RemarkRecognitionResultV2.create_unrecognized(
                    library_id=lib_id,
                    original_text=remark_text,
                    reason=f"未识别退回: 处理异常 - {type(result).__name__}",
                )
            else:
                results[lib_id] = result

        recognized_count = sum(1 for r in results.values() if r.is_recognized)
        unrecognized_count = len(results) - recognized_count
        logger.info(f"批量识别完成 - 成功: {recognized_count}, 未识别: {unrecognized_count}")
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "client_stats": self.client.get_stats(),
            "cache_size": len(self._cache),
            "cache_enabled": self.cache_enabled,
            "confidence_threshold": self.confidence_threshold,
        }
