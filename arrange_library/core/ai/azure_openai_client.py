"""
Azure OpenAI客户端 - 备注识别专用
支持Azure OpenAI API格式
创建时间：2025-11-17
更新时间：2025-12-01 16:45:00
"""
import time
import asyncio
import httpx
from typing import Dict, Optional, Any, List
from loguru import logger
import os

from liblane_paths import setup_liblane_paths
setup_liblane_paths()


class AzureOpenAIClient:
    """Azure OpenAI API客户端 - 备注识别专用"""
    
    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = None,
        model: str = "gpt-4o",
        timeout: int = 30
    ):
        """
        初始化Azure OpenAI客户端
        
        Args:
            api_key: API密钥（从环境变量AZURE_OPENAI_API_KEY获取）
            endpoint: 端点URL（从环境变量AZURE_OPENAI_ENDPOINT获取）
            api_version: API版本（从环境变量OPENAI_API_VERSION获取）
            model: 模型名称
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = api_version or os.getenv('OPENAI_API_VERSION', '2024-08-01-preview')
        self.model = model
        self.timeout = timeout
        self.max_retries = 3
        self.retry_delay = 1.0
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API密钥未配置，请设置环境变量 AZURE_OPENAI_API_KEY")
        if not self.endpoint:
            raise ValueError("Azure OpenAI端点未配置，请设置环境变量 AZURE_OPENAI_ENDPOINT")
        
        # 确保endpoint不以/结尾
        self.endpoint = self.endpoint.rstrip('/')
        
        # 熔断器配置
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = 300  # 5分钟
        self.last_failure_time = 0
        
        # 请求统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info(f"Azure OpenAI客户端初始化完成 - 端点: {self.endpoint}, 模型: {self.model}")
    
    def _is_circuit_breaker_open(self) -> bool:
        """检查熔断器是否开启"""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False
        
        if time.time() - self.last_failure_time > self.circuit_breaker_reset_time:
            self.circuit_breaker_failures = 0
            logger.info("熔断器重置，尝试恢复API调用")
            return False
        
        return True
    
    def _record_success(self):
        """记录成功请求"""
        self.successful_requests += 1
        self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)
    
    def _record_failure(self):
        """记录失败请求"""
        self.failed_requests += 1
        self.circuit_breaker_failures += 1
        self.last_failure_time = time.time()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        调用Azure OpenAI Chat Completions API
        
        Args:
            messages: 消息列表，格式：[{"role": "system", "content": "..."}, ...]
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如{"type": "json_object"}用于JSON模式）
        
        Returns:
            API响应JSON
        """
        if self._is_circuit_breaker_open():
            logger.warning("熔断器开启，跳过API调用")
            raise Exception("Circuit breaker is open")
        
        self.total_requests += 1
        
        # 重置重试延迟，避免状态泄露
        current_retry_delay = 1.0
        
        # Azure OpenAI API URL格式
        # {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}
        url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions"
        params = {"api-version": self.api_version}
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        params=params,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        self._record_success()
                        return response.json()
                    elif response.status_code == 429:
                        logger.warning(f"API请求限流，等待{current_retry_delay}秒后重试")
                        await asyncio.sleep(current_retry_delay)
                        current_retry_delay *= 1.5
                        continue
                    else:
                        error_text = response.text[:500]  # 限制错误信息长度
                        logger.error(f"API请求失败: {response.status_code} - {error_text}")
                        self._record_failure()
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request,
                            response=response
                        )
                        
            except asyncio.TimeoutError:
                logger.warning(f"API请求超时，重试 {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(current_retry_delay)
                    current_retry_delay *= 1.2
                else:
                    self._record_failure()
                    raise
            except Exception as e:
                logger.error(f"API请求异常: {type(e).__name__}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(current_retry_delay)
                    current_retry_delay *= 1.2
                else:
                    self._record_failure()
                    raise
        
        self._record_failure()
        raise Exception(f"API请求失败，已重试{self.max_retries}次")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取请求统计信息"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0.0
            ),
            "circuit_breaker_open": self._is_circuit_breaker_open()
        }

