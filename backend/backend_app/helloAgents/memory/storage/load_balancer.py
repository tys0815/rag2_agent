"""
负载均衡器 - 为向量存储提供负载均衡功能

支持多个向量存储后端，提供负载均衡、故障转移和健康检查。
"""

import time
import random
import threading
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"      # 轮询
    RANDOM = "random"                # 随机
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    PERFORMANCE = "performance"      # 基于性能
    HEALTH_BASED = "health_based"    # 基于健康状态

class BackendStatus:
    """后端状态"""

    def __init__(self, backend_id: str, backend: Any):
        self.backend_id = backend_id
        self.backend = backend
        self.connections = 0  # 当前连接数
        self.total_requests = 0  # 总请求数
        self.successful_requests = 0  # 成功请求数
        self.failed_requests = 0  # 失败请求数
        self.last_response_time = 0.0  # 最后响应时间
        self.avg_response_time = 0.0  # 平均响应时间
        self.last_check_time = 0.0  # 最后检查时间
        self.healthy = True  # 健康状态
        self.last_error = None  # 最后错误信息
        self.weight = 1.0  # 权重（用于加权轮询）

    def update_response_time(self, response_time: float):
        """更新响应时间统计"""
        self.last_response_time = response_time
        if self.total_requests == 0:
            self.avg_response_time = response_time
        else:
            # 指数移动平均
            alpha = 0.2
            self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time

    def record_success(self, response_time: float):
        """记录成功请求"""
        self.connections = max(0, self.connections - 1)
        self.total_requests += 1
        self.successful_requests += 1
        self.update_response_time(response_time)
        self.last_error = None

    def record_failure(self, error: Exception):
        """记录失败请求"""
        self.connections = max(0, self.connections - 1)
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = str(error)

    def start_request(self):
        """开始请求"""
        self.connections += 1

    def get_score(self) -> float:
        """计算后端得分（用于负载均衡）"""
        if not self.healthy:
            return -float('inf')

        # 基于连接数、响应时间和错误率计算得分
        connection_score = 1.0 / (1.0 + self.connections)
        response_score = 1.0 / (1.0 + self.avg_response_time) if self.avg_response_time > 0 else 1.0
        error_rate = self.failed_requests / max(1, self.total_requests)
        error_score = 1.0 - error_rate

        return connection_score * 0.4 + response_score * 0.4 + error_score * 0.2

    def __str__(self) -> str:
        return (f"BackendStatus(id={self.backend_id}, healthy={self.healthy}, "
                f"conns={self.connections}, avg_rt={self.avg_response_time:.3f}s, "
                f"success={self.successful_requests}/{self.total_requests})")

class LoadBalancer:
    """向量存储负载均衡器"""

    def __init__(
        self,
        backends: List[Any],
        backend_ids: Optional[List[str]] = None,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        health_check_interval: float = 30.0,  # 健康检查间隔（秒）
        health_check_timeout: float = 5.0,    # 健康检查超时（秒）
        max_failures: int = 3,                # 最大连续失败次数
        recovery_interval: float = 60.0,      # 恢复检查间隔（秒）
    ):
        """
        初始化负载均衡器

        Args:
            backends: 后端存储实例列表
            backend_ids: 后端ID列表（可选）
            strategy: 负载均衡策略
            health_check_interval: 健康检查间隔（秒）
            health_check_timeout: 健康检查超时（秒）
            max_failures: 最大连续失败次数
            recovery_interval: 恢复检查间隔（秒）
        """
        if not backends:
            raise ValueError("至少需要一个后端")

        self.backends = backends
        self.backend_ids = backend_ids or [f"backend_{i}" for i in range(len(backends))]
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_failures = max_failures
        self.recovery_interval = recovery_interval

        # 初始化后端状态
        self.backend_statuses: Dict[str, BackendStatus] = {}
        for i, (backend_id, backend) in enumerate(zip(self.backend_ids, backends)):
            self.backend_statuses[backend_id] = BackendStatus(backend_id, backend)

        # 负载均衡状态
        self.current_index = 0  # 用于轮询
        self.lock = threading.RLock()

        # 健康检查线程
        self.health_check_thread = None
        self.running = False

        logger.info(f"负载均衡器初始化: {len(backends)} 个后端, 策略={strategy.value}")

    def start(self):
        """启动负载均衡器（开始健康检查）"""
        with self.lock:
            if self.running:
                return

            self.running = True
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="LoadBalancerHealthCheck"
            )
            self.health_check_thread.start()
            logger.info("负载均衡器健康检查已启动")

    def stop(self):
        """停止负载均衡器"""
        with self.lock:
            self.running = False
            if self.health_check_thread:
                self.health_check_thread.join(timeout=5.0)
                self.health_check_thread = None
            logger.info("负载均衡器已停止")

    def _health_check_loop(self):
        """健康检查循环"""
        while self.running:
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")

            time.sleep(self.health_check_interval)

    def _perform_health_checks(self):
        """执行健康检查"""
        with self.lock:
            for backend_id, status in self.backend_statuses.items():
                try:
                    # 检查是否需要健康检查
                    current_time = time.time()
                    if current_time - status.last_check_time < self.health_check_interval:
                        continue

                    status.last_check_time = current_time

                    # 执行健康检查
                    start_time = time.time()
                    healthy = self._check_backend_health(status.backend)
                    response_time = time.time() - start_time

                    if healthy:
                        if not status.healthy:
                            logger.info(f"后端 {backend_id} 已恢复健康")
                        status.healthy = True
                        status.update_response_time(response_time)
                    else:
                        if status.healthy:
                            logger.warning(f"后端 {backend_id} 变为不健康")
                        status.healthy = False

                except Exception as e:
                    logger.error(f"后端 {backend_id} 健康检查失败: {e}")
                    status.healthy = False

    def _check_backend_health(self, backend) -> bool:
        """检查后端健康状态"""
        # 尝试调用后端的健康检查方法
        if hasattr(backend, 'health_check'):
            try:
                return backend.health_check()
            except Exception:
                return False

        # 如果没有健康检查方法，尝试简单操作
        if hasattr(backend, 'get_collection_info'):
            try:
                backend.get_collection_info()
                return True
            except Exception:
                return False

        # 默认认为健康
        return True

    def select_backend(self) -> Optional[Any]:
        """
        选择后端

        Returns:
            选中的后端实例，如果没有可用后端则返回None
        """
        with self.lock:
            # 获取健康的后端
            healthy_backends = [
                (backend_id, status)
                for backend_id, status in self.backend_statuses.items()
                if status.healthy
            ]

            if not healthy_backends:
                logger.error("没有可用的健康后端")
                return None

            # 根据策略选择后端
            if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
                backend_id, status = healthy_backends[self.current_index % len(healthy_backends)]
                self.current_index = (self.current_index + 1) % len(healthy_backends)

            elif self.strategy == LoadBalanceStrategy.RANDOM:
                backend_id, status = random.choice(healthy_backends)

            elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                backend_id, status = min(healthy_backends, key=lambda x: x[1].connections)

            elif self.strategy == LoadBalanceStrategy.PERFORMANCE:
                backend_id, status = max(healthy_backends, key=lambda x: x[1].get_score())

            elif self.strategy == LoadBalanceStrategy.HEALTH_BASED:
                # 基于健康状态和权重
                total_weight = sum(status.weight for _, status in healthy_backends)
                if total_weight <= 0:
                    backend_id, status = random.choice(healthy_backends)
                else:
                    r = random.uniform(0, total_weight)
                    current = 0
                    for bid, stat in healthy_backends:
                        current += stat.weight
                        if r <= current:
                            backend_id, status = bid, stat
                            break
            else:
                backend_id, status = healthy_backends[0]

            # 记录请求开始
            status.start_request()
            return status.backend, backend_id, status

    def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str = "operation",
        max_retries: int = 2,
        retry_delay: float = 1.0,
        fallback_operation: Optional[Callable] = None,
    ) -> Any:
        """
        使用负载均衡和重试执行操作

        Args:
            operation: 要执行的操作函数，接受后端实例作为参数
            operation_name: 操作名称（用于日志）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            fallback_operation: 后备操作（当所有后端都失败时调用）

        Returns:
            操作结果

        Raises:
            Exception: 当所有重试都失败时
        """
        last_error = None
        tried_backends = set()

        for attempt in range(max_retries + 1):
            # 选择后端
            result = self.select_backend()
            if result is None:
                # 没有可用后端，尝试后备操作
                if fallback_operation:
                    try:
                        return fallback_operation()
                    except Exception as e:
                        last_error = e
                        break
                else:
                    raise RuntimeError("没有可用的后端且无后备操作")

            backend, backend_id, status = result

            # 避免重复尝试同一个后端
            if backend_id in tried_backends and len(tried_backends) < len(self.backend_statuses):
                continue

            tried_backends.add(backend_id)

            try:
                # 执行操作
                start_time = time.time()
                result = operation(backend)
                response_time = time.time() - start_time

                # 记录成功
                status.record_success(response_time)
                logger.debug(f"{operation_name} 成功: 后端={backend_id}, 耗时={response_time:.3f}s")
                return result

            except Exception as e:
                # 记录失败
                status.record_failure(e)
                last_error = e
                logger.warning(f"{operation_name} 失败 (尝试 {attempt+1}/{max_retries+1}): "
                             f"后端={backend_id}, 错误={e}")

                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries:
                    time.sleep(retry_delay * (attempt + 1))

        # 所有尝试都失败
        if fallback_operation:
            try:
                logger.warning(f"所有后端失败，尝试后备操作: {operation_name}")
                return fallback_operation()
            except Exception as e:
                last_error = e

        error_msg = f"{operation_name} 失败: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from last_error

    def get_stats(self) -> Dict[str, Any]:
        """获取负载均衡器统计信息"""
        with self.lock:
            stats = {
                "strategy": self.strategy.value,
                "total_backends": len(self.backend_statuses),
                "healthy_backends": sum(1 for s in self.backend_statuses.values() if s.healthy),
                "backend_details": {}
            }

            for backend_id, status in self.backend_statuses.items():
                stats["backend_details"][backend_id] = {
                    "healthy": status.healthy,
                    "connections": status.connections,
                    "total_requests": status.total_requests,
                    "successful_requests": status.successful_requests,
                    "failed_requests": status.failed_requests,
                    "avg_response_time": status.avg_response_time,
                    "last_error": status.last_error,
                    "weight": status.weight
                }

            return stats

    def set_backend_weight(self, backend_id: str, weight: float):
        """设置后端权重"""
        with self.lock:
            if backend_id in self.backend_statuses:
                self.backend_statuses[backend_id].weight = max(0.0, weight)
                logger.info(f"后端 {backend_id} 权重设置为 {weight}")
            else:
                raise ValueError(f"未知的后端ID: {backend_id}")

    def mark_backend_unhealthy(self, backend_id: str, reason: str = ""):
        """标记后端为不健康"""
        with self.lock:
            if backend_id in self.backend_statuses:
                self.backend_statuses[backend_id].healthy = False
                logger.warning(f"标记后端 {backend_id} 为不健康: {reason}")
            else:
                raise ValueError(f"未知的后端ID: {backend_id}")

    def mark_backend_healthy(self, backend_id: str):
        """标记后端为健康"""
        with self.lock:
            if backend_id in self.backend_statuses:
                self.backend_statuses[backend_id].healthy = True
                logger.info(f"标记后端 {backend_id} 为健康")
            else:
                raise ValueError(f"未知的后端ID: {backend_id}")

class LoadBalancedVectorStore:
    """负载均衡的向量存储包装器"""

    def __init__(
        self,
        backends: List[Any],
        backend_ids: Optional[List[str]] = None,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        **kwargs
    ):
        """
        初始化负载均衡向量存储

        Args:
            backends: 后端向量存储实例列表
            backend_ids: 后端ID列表
            strategy: 负载均衡策略
            **kwargs: 传递给LoadBalancer的额外参数
        """
        self.load_balancer = LoadBalancer(
            backends=backends,
            backend_ids=backend_ids,
            strategy=strategy,
            **kwargs
        )

        # 启动负载均衡器
        self.load_balancer.start()

    def __getattr__(self, name):
        """将方法调用路由到负载均衡器"""
        def method(*args, **kwargs):
            def operation(backend):
                return getattr(backend, name)(*args, **kwargs)

            return self.load_balancer.execute_with_retry(
                operation=operation,
                operation_name=f"{name}"
            )

        return method

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """获取负载均衡器统计信息"""
        return self.load_balancer.get_stats()

    def stop(self):
        """停止负载均衡器"""
        self.load_balancer.stop()

# 便捷函数
def create_load_balanced_store(
    urls: List[str],
    api_keys: Optional[List[str]] = None,
    collection_name: str = "hello_agents_vectors",
    vector_size: int = 384,
    distance: str = "cosine",
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    **kwargs
) -> LoadBalancedVectorStore:
    """
    创建负载均衡的向量存储

    Args:
        urls: Qdrant URL列表
        api_keys: API密钥列表（可选）
        collection_name: 集合名称
        vector_size: 向量维度
        distance: 距离度量方式
        strategy: 负载均衡策略
        **kwargs: 传递给LoadBalancer的额外参数

    Returns:
        负载均衡向量存储实例
    """
    from .qdrant_store import QdrantVectorStore

    if api_keys is None:
        api_keys = [None] * len(urls)

    if len(api_keys) != len(urls):
        raise ValueError("URLs和API密钥数量不匹配")

    backends = []
    for i, (url, api_key) in enumerate(zip(urls, api_keys)):
        backend = QdrantVectorStore(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance
        )
        backends.append(backend)

    return LoadBalancedVectorStore(
        backends=backends,
        backend_ids=[f"qdrant_{i}" for i in range(len(urls))],
        strategy=strategy,
        **kwargs
    )