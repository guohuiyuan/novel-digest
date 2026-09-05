"""
api_client.py

统一管理：调用 API 时的错误码处理、错误信息提取、超时识别、API_KEY 池轮换与智能重试。

目标：其他脚本只需要 `from api_client import make_chat_completion`，
并用工厂函数生成 `chat_completion`，从而做到只修改本文件即可全局生效。

=== 关键设计说明（为什么这样做）===

1. 【为什么必须关闭 SDK 暗重试】
   openai SDK 和底层 httpx 默认有自动重试机制。如果不关闭，外层重试 + 内层重试会叠加，
   导致单次请求可能耗时 120s * 3(内层) * 5(外层) = 1800s，严重浪费时间。
   日志中 "Retrying request to /chat/completions" 就是 SDK 暗重试的证据。

2. 【为什么要做本地 RPM/TPM 预限流】
   厂商在触发限速时，可能不返回 429/403 错误码，而是"挂起排队"——请求卡住等待，
   导致超时被误判为 key 失效。本地预限流可以在发请求前就控制速率，避免打到厂商阈值。

3. 【为什么要动态 timeout】
   大输出（如 max_tokens=4000）的生成时间可能远超默认 120s，固定 timeout 会把
   "正常慢生成"误判为"key 挂起"，导致好 key 被禁用。动态 timeout 根据请求规模放宽。

4. 【为什么"无可用 key"要等待而不是直接报错】
   如果所有 key 都只是"暂时软禁用"（如限速冷却），直接报错会导致 chunk 连锁失败。
   正确做法是等待最早解禁时间，然后继续尝试。只有全部永久禁用才真正报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import threading
import hashlib
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from collections import deque


def extract_status_code(err: BaseException) -> Optional[int]:
    """尽力从异常对象中提取 HTTP 状态码（兼容 openai/httpx/requests 风格异常）。"""
    for attr in ("status_code", "status", "http_status"):
        code = getattr(err, attr, None)
        if isinstance(code, int):
            return code
    resp = getattr(err, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def extract_error_message(err: BaseException) -> str:
    """尽力从异常对象中提取错误信息文本。"""
    resp = getattr(err, "response", None)
    if resp is not None:
        try:
            body = resp.json() if hasattr(resp, "json") else None
            if body:
                msg = body.get("error", {}).get("message", "") or body.get("message", "")
                if msg:
                    return str(msg)
        except Exception:
            pass
    return str(err)


def is_timeout_error(err: BaseException) -> bool:
    """判断是否为超时/连接类错误（部分 SDK 不带状态码）。"""
    err_str = str(err).lower()
    err_type = type(err).__name__.lower()
    timeout_keywords = ("timeout", "timed out", "超时", "connection", "connecterror", "readtimeout", "writetimeout")
    return any(kw in err_str or kw in err_type for kw in timeout_keywords)


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 2.0
    max_403_retries: int = 3
    max_timeout_retries: int = 3
    request_timeout: int = 120


def _log(logger: Any, level: str, msg: str) -> None:
    """logger 既支持 logging.Logger，也支持 None（退化为 print）。"""
    try:
        if logger is None:
            print(msg)
            return
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(msg)
            return
    except Exception:
        pass
    # 兜底
    print(msg)


def _key_fingerprint(key: str) -> str:
    """
    生成脱敏 key 标识：hash 前 8 位 + 后 4 位。
    形如：a1b2c3d4|3d05
    """
    if not key:
        return "empty"
    try:
        h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:8]
    except Exception:
        h = "????????"
    tail4 = key[-4:] if len(key) >= 4 else key
    return f"{h}|{tail4}"


# ===================== 本地 RPM/TPM 限流器 =====================
class RateLimiter:
    """Sliding-window limiter supporting global/per-key scopes."""

    GLOBAL_SCOPE = "GLOBAL_SCOPE"

    def __init__(
        self,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        window_seconds: int = 60,
        scope: str = "global",
    ):
        env_rpm = os.environ.get("RPM_LIMIT")
        default_rpm = 60
        if env_rpm:
            try:
                default_rpm = int(env_rpm.strip())
            except ValueError:
                pass

        env_tpm = os.environ.get("TPM_LIMIT")
        default_tpm = 100000
        if env_tpm:
            try:
                default_tpm = int(env_tpm.strip())
            except ValueError:
                pass

        normalized_scope = str(scope or "").strip().lower()
        if normalized_scope not in ("global", "per_key"):
            normalized_scope = "global"

        self.rpm_limit = rpm_limit if rpm_limit is not None else default_rpm
        self.tpm_limit = tpm_limit if tpm_limit is not None else default_tpm
        self.window_seconds = window_seconds
        self.scope = normalized_scope
        self._lock = threading.Lock()
        # bucket -> deque of (timestamp, tokens)
        self._requests: Dict[str, deque] = {}

    @property
    def is_enabled(self) -> bool:
        return self.rpm_limit > 0 or self.tpm_limit > 0

    def _bucket_for(self, key: str) -> str:
        return self.GLOBAL_SCOPE if self.scope == "global" else key

    def _cleanup(self, bucket: str, now: float) -> None:
        if bucket not in self._requests:
            return
        cutoff = now - self.window_seconds
        q = self._requests[bucket]
        while q and q[0][0] < cutoff:
            q.popleft()

    def _get_current_usage(self, bucket: str, now: float) -> Tuple[int, int]:
        self._cleanup(bucket, now)
        if bucket not in self._requests:
            return (0, 0)
        q = self._requests[bucket]
        rpm = len(q)
        tpm = sum(tokens for _, tokens in q)
        return (rpm, tpm)

    def _compute_delay_locked(self, bucket: str, est_tokens: int, now: float) -> Tuple[float, str]:
        rpm, tpm = self._get_current_usage(bucket, now)
        q = self._requests.get(bucket)
        oldest_ts = q[0][0] if q else now

        wait_rpm = 0.0
        if self.rpm_limit > 0 and rpm >= self.rpm_limit:
            wait_rpm = max(0.0, oldest_ts + self.window_seconds - now + 0.1)

        wait_tpm = 0.0
        if self.tpm_limit > 0 and (tpm + est_tokens) > self.tpm_limit:
            wait_tpm = max(0.0, oldest_ts + self.window_seconds - now + 0.1)

        if wait_rpm > 0 and wait_tpm > 0:
            reason = "rpm+tpm"
        elif wait_rpm > 0:
            reason = "rpm"
        elif wait_tpm > 0:
            reason = "tpm"
        else:
            reason = "ok"

        return (max(wait_rpm, wait_tpm), reason)

    def preview_delay_for(self, key: str, est_tokens: int) -> Tuple[float, str]:
        if not self.is_enabled:
            return (0.0, "disabled")
        bucket = self._bucket_for(key)
        with self._lock:
            return self._compute_delay_locked(bucket, est_tokens, time.time())

    def acquire_slot(self, key: str, est_tokens: int) -> Tuple[float, str]:
        """Atomic check-and-wait + reserve."""
        if not self.is_enabled:
            return (0.0, "disabled")
        bucket = self._bucket_for(key)
        with self._lock:
            now = time.time()
            delay, reason = self._compute_delay_locked(bucket, est_tokens, now)
            if delay > 0:
                return (delay, reason)
            self._requests.setdefault(bucket, deque()).append((now, est_tokens))
            return (0.0, "ok")


# ===================== IP 级别全局冷却管理器 =====================
class IPCooldownManager:
    """
    IP 级别全局冷却。当任意 key 触发限速（429/403 RPM/TPM）时，所有请求暂停。

    【为什么需要 IP 级冷却】
    API 厂商的限速可能基于 IP 而非 API_KEY。当一个 key 触发限速后，
    切换到另一个 key 并不能绕过限速——因为它们共享同一个 IP。
    如果不做全局暂停，所有 key 的重试都会持续产生请求，
    导致 IP 限速永远无法解除，最终所有 key 被耗尽。

    冷却策略：指数递增 30s → 60s → 120s → 240s → 300s(cap)
    防抖机制：5 秒内多个线程的重复触发只算一次惩罚
    """

    def __init__(self, base_cooldown: float = 30.0, max_cooldown: float = 300.0):
        self._lock = threading.Lock()
        self._cooldown_until: float = 0.0
        self._consecutive_triggers: int = 0
        self._last_trigger_ts: float = 0.0
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown

    def trigger(self, reason: str = "") -> float:
        """
        触发 IP 冷却，返回冷却剩余秒数。
        【防抖】5 秒内多次调用只递增一次惩罚计数，
        防止 7-12 个并发线程同时触发导致瞬间冲到顶格冷却。
        """
        with self._lock:
            now = time.time()
            # 防抖：同一批并发请求导致的报错，5秒内不重复叠加惩罚基数
            if now - self._last_trigger_ts > 5.0:
                self._consecutive_triggers += 1
                self._last_trigger_ts = now

            # 计算冷却，至少为 1 次
            power = max(0, self._consecutive_triggers - 1)
            cooldown = min(self.base_cooldown * (2 ** power), self.max_cooldown)
            jitter = random.uniform(1.0, 3.0)
            target = now + cooldown + jitter

            if target > self._cooldown_until:
                self._cooldown_until = target

            return max(0.0, self._cooldown_until - time.time())

    def trigger_micro_cooldown(self, seconds: float) -> None:
        """
        微冷却：设置短暂冷却（如 1.5s），不递增惩罚计数。
        用于 key 永久禁用后防止并发线程瞬间涌入下一个 key。
        带随机抖动使并发线程醒来时间打散，配合 RateLimiter 平滑过渡。
        如果当前已有更长的冷却，则不覆盖。
        """
        with self._lock:
            jitter = random.uniform(0.1, 0.5)
            target = time.time() + seconds + jitter
            if target > self._cooldown_until:
                self._cooldown_until = target

    def wait_if_needed(self, logger: Any = None) -> float:
        """如果冷却中则阻塞等待，返回实际等待秒数。"""
        with self._lock:
            remaining = self._cooldown_until - time.time()
        if remaining <= 0:
            return 0.0
        _log(
            logger,
            "warning",
            f"IP全局冷却中，等待 {remaining:.1f}s（连续触发{self._consecutive_triggers}次）",
        )
        time.sleep(remaining)
        return remaining

    def on_success(self) -> None:
        """请求成功，重置连续触发计数（下次限速从 base_cooldown 重新开始）。"""
        with self._lock:
            self._consecutive_triggers = 0

    def has_recent_rate_limit(self, within_seconds: float = 60.0) -> bool:
        """最近 within_seconds 秒内是否有限速触发（且未被成功请求重置）。"""
        with self._lock:
            if self._last_trigger_ts == 0.0:
                return False
            # 必须同时满足：时间差以内 + 连续触发次数未被 on_success 清零
            return (time.time() - self._last_trigger_ts) <= within_seconds and self._consecutive_triggers > 0


# ===================== Token 估算 =====================
def estimate_tokens(messages: Any, max_tokens: Optional[int] = None) -> int:
    """
    粗略估算本次请求的 token 消耗（输入 + 输出）。
    
    估算方法：
    - 中文：约 2 字符/token
    - 英文：约 4 字符/token
    - 混合取 3 字符/token
    
    如果环境有 tiktoken 可以更精确，但这里不引入强依赖。
    """
    # 估算输入 tokens
    input_chars = 0
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    input_chars += len(content)
                elif isinstance(content, list):
                    # 多模态消息
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            input_chars += len(part.get("text", ""))
    
    # 粗略估算：3 字符 ≈ 1 token
    input_tokens = max(1, input_chars // 3)
    
    # 输出 tokens：取 max_tokens 或默认 2000
    output_tokens = max_tokens if max_tokens else 2000
    
    return input_tokens + output_tokens


def compute_dynamic_timeout(
    base_timeout: int,
    max_tokens: Optional[int],
    input_chars: int,
    alpha: float = 0.08,
    cap: int = 300,
) -> int:
    """
    计算动态 timeout。
    
    公式：dyn_timeout = max(base_timeout, 40 + alpha * max_tokens)
    并 cap 到最大值。
    
    参数：
    - base_timeout: 基础超时（如 120s）
    - max_tokens: 最大输出 token 数
    - input_chars: 输入字符数（用于大输入场景）
    - alpha: 每 token 增加的秒数（默认 0.08）
    - cap: 最大超时上限（默认 300s）
    """
    # 基于 max_tokens 计算
    if max_tokens:
        computed = 40 + alpha * max_tokens
    else:
        computed = 40 + alpha * 2000  # 默认假设 2000 tokens
    
    # 大输入也需要更长时间（每 10000 字符加 10 秒）
    computed += (input_chars // 10000) * 10
    
    # 取 max 并 cap
    dyn_timeout = max(base_timeout, int(computed))
    dyn_timeout = min(dyn_timeout, cap)
    
    return dyn_timeout


# ===================== 示例 openai_client_factory =====================
def create_openai_client_factory_no_retry():
    """
    创建一个关闭 SDK 暗重试、使用细粒度 timeout 的 openai_client_factory。
    
    【关键】必须设置 max_retries=0 关闭 SDK 自动重试，否则：
    - SDK 默认会重试 2 次
    - 每次重试都有 timeout
    - 外层 api_client.py 再重试 5 次
    - 总耗时可能达到 120s * 3 * 5 = 1800s
    
    【关键】使用 httpx.Timeout 细粒度配置：
    - connect: 连接超时（通常较短，如 10s）
    - read: 读取超时（需要根据请求规模动态调整）
    - write: 写入超时（通常较短，如 30s）
    - pool: 连接池超时（通常较短，如 10s）
    """
    try:
        import httpx
        HAS_HTTPX = True
    except ImportError:
        HAS_HTTPX = False
    
    def factory(api_key: str, base_url: str, timeout: int):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请安装 openai: pip install openai")
        
        if HAS_HTTPX:
            # 使用细粒度 timeout
            # connect: 建立连接的超时
            # read: 等待服务器响应的超时（这是最重要的，需要动态调整）
            # write: 发送请求的超时
            # pool: 从连接池获取连接的超时
            http_timeout = httpx.Timeout(
                connect=10.0,
                read=float(timeout),  # 读取超时使用传入的 timeout
                write=30.0,
                pool=10.0,
            )
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=http_timeout,
                max_retries=0,  # 【关键】关闭 SDK 自动重试
            )
        else:
            # 没有 httpx 时使用简单 timeout
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,  # 【关键】关闭 SDK 自动重试
            )
    
    return factory


# 默认的 factory（关闭暗重试）
default_openai_client_factory = create_openai_client_factory_no_retry()


def make_chat_completion(
    *,
    openai_client_factory: Callable[[str, str, int], Any],
    api_key_pool: Sequence[str],
    base_url: str,
    request_timeout: int = 120,
    max_retries: int = 5,
    max_403_retries: int = 3,
    max_timeout_retries: int = 3,
    base_delay: float = 2.0,
    # -------- 超时/慢响应的区分与误杀保护 --------
    rate_limit_grace_seconds: int = 180,
    timeout_soft_disable_base_seconds: int = 45,
    timeout_permanent_disable_after: int = 20,
    recent_success_protect_seconds: int = 1800,
    # -------- 本地 RPM/TPM 限流（可选） --------
    rpm_limit: Optional[int] = None,  # 每分钟请求数限制，None 表示不限流
    tpm_limit: Optional[int] = None,  # 每分钟 token 数限制，None 表示不限流
    rate_limit_scope: str = "global",  # global / per_key
    # -------- 动态 timeout 配置 --------
    dynamic_timeout_alpha: float = 0.08,  # 每 token 增加的秒数
    dynamic_timeout_cap: int = 300,  # 最大超时上限
    logger: Any = None,
) -> Callable[..., Any]:
    """
    工厂：生成一个带有独立状态（禁用 key / 连续错误计数 / 限流器）的 chat_completion 函数。

    这样每个脚本只需：
    - 定义自己的 API_KEY_POOL/BASE_URL/REQUEST_TIMEOUT/重试阈值
    - `chat_completion = make_chat_completion(...)`
    后续统一修改本文件即可影响所有脚本的错误处理与重试策略。
    
    新增功能：
    1. 本地 RPM/TPM 预限流（避免打到厂商阈值被挂起）
    2. 动态 timeout（根据请求规模放宽 timeout，防止大输出被误判）
    3. 等待最早解禁（避免所有 key 暂时不可用时直接报错）
    4. 改进的软禁用/永久禁用判定（保护近期成功的 key）
    """
    cfg = RetryConfig(
        max_retries=int(max_retries),
        base_delay=float(base_delay),
        max_403_retries=int(max_403_retries),
        max_timeout_retries=int(max_timeout_retries),
        request_timeout=int(request_timeout),
    )

    # 限流器（始终启用，防止冷启动时多线程同时涌入触发 IP 限速）
    # 优先级：函数参数 > 环境变量 > 默认值（RPM=60, TPM=100000）
    # 用户传入 rpm_limit=0 可显式关闭限流
    env_scope = str(os.environ.get("RATE_LIMIT_SCOPE", "")).strip().lower()
    if env_scope not in ("global", "per_key"):
        env_scope = ""
    param_scope = str(rate_limit_scope or "").strip().lower()
    if param_scope not in ("global", "per_key"):
        param_scope = ""
    # Keep env usable for existing call sites that don't pass new argument.
    resolved_scope = param_scope or env_scope or "global"

    rate_limiter = RateLimiter(
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        scope=resolved_scope,
    )
    _log(
        logger,
        "info",
        f"本地限流配置：enabled={rate_limiter.is_enabled}, scope={rate_limiter.scope}, "
        f"rpm={rate_limiter.rpm_limit}, tpm={rate_limiter.tpm_limit}, window={rate_limiter.window_seconds}s",
    )
    if not rate_limiter.is_enabled:
        _log(logger, "info", "本地限流已关闭（RPM_LIMIT<=0 且 TPM_LIMIT<=0）")

    # IP 级别全局冷却（与 disabled_keys_lock 同级，作为闭包变量被所有工作线程共享）
    ip_cooldown = IPCooldownManager(base_cooldown=30.0, max_cooldown=300.0)

    disabled_keys_lock = threading.Lock()
    permanently_disabled: set[str] = set()
    disabled_until: dict[str, float] = {}

    counts_lock = threading.Lock()
    key_403_counts: dict[str, int] = {}
    key_timeout_counts: dict[str, int] = {}
    key_success_counts: dict[str, int] = {}
    key_last_success_ts: dict[str, float] = {}
    key_last_rate_limit_ts: dict[str, float] = {}

    def _get_available_keys() -> Tuple[List[str], Optional[float]]:
        """
        获取当前可用的 key 列表。
        返回 (available_keys, earliest_unblock_time)
        - available_keys: 当前可用的 key 列表
        - earliest_unblock_time: 如果没有可用 key，返回最早解禁时间；否则返回 None
        """
        now = time.time()
        available = []
        earliest_unblock = None
        
        with disabled_keys_lock:
            for k in api_key_pool:
                if not k:
                    continue
                if k in permanently_disabled:
                    continue
                until = disabled_until.get(k, 0.0)
                if until and until > now:
                    # 记录最早解禁时间
                    if earliest_unblock is None or until < earliest_unblock:
                        earliest_unblock = until
                    continue
                available.append(k)
        
        return (available, earliest_unblock)

    def _soft_disable_key(key: str, cooldown: float) -> None:
        """软禁用 key 一段时间（带 jitter 避免惊群）"""
        jitter = random.uniform(0, cooldown * 0.2)  # 0-20% 的 jitter
        with disabled_keys_lock:
            disabled_until[key] = time.time() + cooldown + jitter

    def _permanent_disable_key(key: str) -> None:
        """永久禁用 key"""
        with disabled_keys_lock:
            permanently_disabled.add(key)
            disabled_until.pop(key, None)

    def _calculate_input_chars(messages: Any) -> int:
        """计算输入消息的总字符数"""
        total = 0
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        total += len(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                total += len(part.get("text", ""))
        return total

    def chat_completion(*, messages: Any, **kwargs: Any) -> Any:
        if not api_key_pool:
            raise ValueError("请设置 API_KEY 或 API_KEY_POOL")

        # 计算动态 timeout
        max_tokens = kwargs.get("max_tokens")
        # 结构化输出（response_format=json_object）若未显式给输出预算，推理模型会超长思考，
        # 在该服务上表现为 200s 挂起。统一补一个默认输出上限，避免此类超时。
        if max_tokens is None:
            _rf = kwargs.get("response_format")
            if isinstance(_rf, dict) and _rf.get("type") == "json_object":
                max_tokens = 1200
                kwargs["max_tokens"] = max_tokens
        input_chars = _calculate_input_chars(messages)
        dyn_timeout = compute_dynamic_timeout(
            base_timeout=cfg.request_timeout,
            max_tokens=max_tokens,
            input_chars=input_chars,
            alpha=dynamic_timeout_alpha,
            cap=dynamic_timeout_cap,
        )
        
        # 估算 tokens（用于限流）
        est_tokens = estimate_tokens(messages, max_tokens)

        outer_attempts = 0
        max_outer_attempts = cfg.max_retries * len(api_key_pool) + 10  # 防止无限循环

        while outer_attempts < max_outer_attempts:
            outer_attempts += 1

            # 【第一步】IP 冷却等待 —— 必须在任何状态计算之前
            ip_cooldown.wait_if_needed(logger)

            # 【第二步】冷却结束后才获取最新时间和 key 列表
            now = time.time()
            break_to_outer = False

            available_keys, earliest_unblock = _get_available_keys()
            
            # 【关键改进】如果没有可用 key，检查是否有软禁用的 key 即将解禁
            if not available_keys:
                # 检查是否所有 key 都永久禁用
                with disabled_keys_lock:
                    all_permanent = all(
                        k in permanently_disabled
                        for k in api_key_pool if k
                    )
                
                if all_permanent or not api_key_pool:
                    raise RuntimeError("所有 API_KEY 均永久不可用（余额不足/无效）")
                
                # 有软禁用的 key，等待最早解禁
                if earliest_unblock is not None:
                    wait_time = max(0, earliest_unblock - now) + 0.5  # 加 0.5s 缓冲
                    _log(
                        logger,
                        "warning",
                        f"当前无可用 API_KEY，将等待 {wait_time:.1f}s 后重试（最早解禁时间：{time.strftime('%H:%M:%S', time.localtime(earliest_unblock))}）",
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # 理论上不应该到这里
                    raise RuntimeError("所有 API_KEY 均不可用")

            # 按预估等待时间排序（稳定排序，delay 相同保持原顺序）
            key_delay_preview: List[Tuple[str, float, str]] = []
            for k in available_keys:
                d, r = rate_limiter.preview_delay_for(k, est_tokens)
                key_delay_preview.append((k, d, r))
            key_delay_preview.sort(key=lambda x: x[1])
            available_keys = [k for k, _, _ in key_delay_preview]

            for idx, key in enumerate(available_keys):
                key_tag = _key_fingerprint(key)

                real_attempt = 0
                while real_attempt < cfg.max_retries:
                    delay, rl_reason = rate_limiter.acquire_slot(key, est_tokens)
                    if delay > 0:
                        _log(
                            logger,
                            "info",
                            f"限流等待：key[{key_tag}] 需等待 {delay:.1f}s（reason={rl_reason}, scope={rate_limiter.scope}）",
                        )
                        time.sleep(delay)
                        break_to_outer = True
                        break

                    # 只有真实发起出站请求时才消耗 attempt
                    real_attempt += 1
                    attempt_no = real_attempt
                    start_ts = time.time()
                    try:
                        # 使用动态 timeout 创建客户端
                        cli = openai_client_factory(key, base_url, dyn_timeout)
                        
                        _log(
                            logger,
                            "info",
                            f"API_KEY[{idx}|{key_tag}] 发起请求：attempt={attempt_no}/{cfg.max_retries}, dyn_timeout={dyn_timeout}s, max_tokens={max_tokens}, input_chars={input_chars}, est_tokens={est_tokens}",
                        )
                        
                        # 针对推理模型（如 qwen3 系列，由 SGLang 提供）：
                        # 关闭 thinking，避免推理过程挤占 max_tokens 预算导致 JSON 被截断，
                        # 同时显著加快出结果。通过环境变量 DISABLE_THINKING 开启（默认关闭，不影响其他配置）。
                        call_kwargs = dict(kwargs)
                        if os.environ.get("DISABLE_THINKING", "").strip().lower() in ("1", "true", "yes"):
                            extra = dict(call_kwargs.get("extra_body") or {})
                            ctk = dict(extra.get("chat_template_kwargs") or {})
                            ctk.setdefault("enable_thinking", False)
                            extra["chat_template_kwargs"] = ctk
                            call_kwargs["extra_body"] = extra

                        resp = cli.chat.completions.create(messages=messages, **call_kwargs)
                        
                        elapsed = time.time() - start_ts
                        
                        # 成功：重置计数，记录成功时间
                        with counts_lock:
                            key_403_counts[key] = 0
                            key_timeout_counts[key] = 0
                            key_success_counts[key] = key_success_counts.get(key, 0) + 1
                            key_last_success_ts[key] = time.time()

                        # 重置 IP 冷却递增计数（下次限速从 base_cooldown 重新开始）
                        ip_cooldown.on_success()
                        
                        # 成功意味着 key 可用，清理软禁用
                        with disabled_keys_lock:
                            if key in disabled_until:
                                disabled_until.pop(key, None)
                        
                        _log(
                            logger,
                            "info",
                            f"API_KEY[{idx}|{key_tag}] 请求成功，耗时 {elapsed:.1f}s",
                        )
                        
                        return resp
                        
                    except Exception as e:  # noqa: BLE001 - 需要兼容多种 SDK/异常类型
                        code = extract_status_code(e)
                        err_msg = extract_error_message(e)
                        elapsed = time.time() - start_ts

                        # 获取 key 状态信息（用于判断）
                        with counts_lock:
                            last_rl = key_last_rate_limit_ts.get(key, 0.0)
                            last_ok = key_last_success_ts.get(key, 0.0)
                            success_cnt = key_success_counts.get(key, 0)
                        
                        now2 = time.time()
                        recently_rate_limited = (now2 - last_rl) <= float(rate_limit_grace_seconds) if last_rl else False
                        recently_succeeded = (now2 - last_ok) <= float(recent_success_protect_seconds) if last_ok else False

                        # ========== 超时处理 ==========
                        if code is None and is_timeout_error(e):
                            with counts_lock:
                                key_timeout_counts[key] = key_timeout_counts.get(key, 0) + 1
                                consecutive_timeout = key_timeout_counts[key]

                            # 【改进】区分短超时（轻微抖动）和长挂起
                            is_long_hang = elapsed >= dyn_timeout * 0.8  # 接近 dyn_timeout 视为长挂起
                            
                            # 记录详细日志
                            _log(
                                logger,
                                "warning",
                                f"API_KEY[{idx}|{key_tag}] 超时：attempt={attempt_no}/{cfg.max_retries}, "
                                f"elapsed={elapsed:.1f}s, dyn_timeout={dyn_timeout}s, "
                                f"consecutive={consecutive_timeout}, "
                                f"recently_rate_limited={recently_rate_limited}, recently_succeeded={recently_succeeded}, "
                                f"is_long_hang={is_long_hang}",
                            )

                            # 长挂起：立即软禁用并切换 key
                            if is_long_hang:
                                cooldown = float(timeout_soft_disable_base_seconds) * max(1, consecutive_timeout)
                                _soft_disable_key(key, cooldown)
                                _log(
                                    logger,
                                    "warning",
                                    f"API_KEY[{idx}|{key_tag}] 长时间挂起({elapsed:.1f}s)，软禁用 {cooldown:.0f}s，切换下一个 key",
                                )
                                break  # 换下一个 key

                            # 达到超时阈值：软禁用
                            if consecutive_timeout >= cfg.max_timeout_retries:
                                cooldown = float(timeout_soft_disable_base_seconds) * max(1, consecutive_timeout)
                                _soft_disable_key(key, cooldown)

                                # 【改进】永久禁用判定更保守
                                # 只有：从未成功过 + 长时间持续超时 + 最近无 rate-limit 迹象
                                can_permanent = (
                                    not recently_rate_limited
                                    and not recently_succeeded
                                    and success_cnt == 0
                                    and consecutive_timeout >= int(timeout_permanent_disable_after)
                                )
                                
                                if can_permanent:
                                    _permanent_disable_key(key)
                                    ip_cooldown.trigger_micro_cooldown(1.5)
                                    _log(
                                        logger,
                                        "warning",
                                        f"API_KEY[{idx}|{key_tag}] 长期超时(累计{consecutive_timeout}次)，"
                                        f"从未成功过，疑似余额不足/挂起，永久禁用",
                                    )
                                else:
                                    hint = ""
                                    if recently_rate_limited:
                                        hint = "（近期限速，疑似慢响应）"
                                    elif recently_succeeded:
                                        hint = "（近期成功过，防误杀）"
                                    _log(
                                        logger,
                                        "warning",
                                        f"API_KEY[{idx}|{key_tag}] 超时累计{consecutive_timeout}次，"
                                        f"软禁用 {cooldown:.0f}s{hint}",
                                    )
                                break  # 换下一个 key

                            # 短超时：线性退避后重试
                            wait_time = cfg.base_delay * attempt_no
                            if recently_rate_limited:
                                wait_time = max(wait_time, cfg.base_delay * 4 + (attempt_no - 1) * 2)
                            _log(
                                logger,
                                "warning",
                                f"API_KEY[{idx}|{key_tag}] 短超时({elapsed:.1f}s)，"
                                f"等待 {wait_time:.1f}s 后重试 ({attempt_no}/{cfg.max_retries})",
                            )
                            time.sleep(wait_time)
                            continue

                        # ========== 401：key 无效 ==========
                        if code == 401:
                            _permanent_disable_key(key)
                            ip_cooldown.trigger_micro_cooldown(1.5)
                            _log(
                                logger,
                                "warning",
                                f"API_KEY[{idx}|{key_tag}] 调用失败(401: API key 无效)，永久禁用",
                            )
                            break_to_outer = True
                            break

                        # ========== 429：速率限制 ==========
                        if code == 429:
                            with counts_lock:
                                key_last_rate_limit_ts[key] = time.time()
                            cooldown_secs = ip_cooldown.trigger(reason=f"429 on key[{key_tag}]")
                            _log(
                                logger,
                                "warning",
                                f"API_KEY[{idx}|{key_tag}] 触发速率限制(429)，"
                                f"IP全局冷却 {cooldown_secs:.1f}s",
                            )
                            break_to_outer = True
                            break

                        # ========== 403：可能是频率限制/余额不足 ==========
                        if code == 403:
                            err_msg_lower = err_msg.lower()
                            
                            is_rate_limit = any(
                                kw in err_msg_lower
                                for kw in (
                                    "rpm limit",
                                    "tpm limit",
                                    "rate limit",
                                    "limit exceeded",
                                    "too many requests",
                                    "请求过于频繁",
                                    "频率限制",
                                )
                            )
                            if is_rate_limit:
                                with counts_lock:
                                    key_last_rate_limit_ts[key] = time.time()
                                cooldown_secs = ip_cooldown.trigger(reason=f"403 RPM/TPM on key[{key_tag}]")
                                _log(
                                    logger,
                                    "warning",
                                    f"API_KEY[{idx}|{key_tag}] 触发速率限制(403: RPM/TPM limit)，"
                                    f"IP全局冷却 {cooldown_secs:.1f}s",
                                )
                                break_to_outer = True
                                break

                            is_balance_issue = any(
                                kw in err_msg_lower
                                for kw in ("balance", "insufficient", "quota", "余额", "不足", "欠费", "充值")
                            )
                            if is_balance_issue:
                                _permanent_disable_key(key)
                                ip_cooldown.trigger_micro_cooldown(1.5)
                                _log(
                                    logger,
                                    "warning",
                                    f"API_KEY[{idx}|{key_tag}] 调用失败(403: 余额不足)，永久禁用",
                                )
                                break_to_outer = True
                                break

                            # 其他 403
                            with counts_lock:
                                key_403_counts[key] = key_403_counts.get(key, 0) + 1
                                consecutive_403 = key_403_counts[key]

                            if consecutive_403 >= cfg.max_403_retries:
                                _permanent_disable_key(key)
                                ip_cooldown.trigger_micro_cooldown(1.5)
                                _log(
                                    logger,
                                    "warning",
                                    f"API_KEY[{idx}|{key_tag}] 调用失败(403: 连续{consecutive_403}次错误)，永久禁用",
                                )
                                break_to_outer = True
                                break

                            wait_time = cfg.base_delay * attempt_no
                            _log(
                                logger,
                                "warning",
                                f"API_KEY[{idx}|{key_tag}] 收到403({err_msg[:80]})，"
                                f"等待 {wait_time:.1f}s 后重试 ({attempt_no}/{cfg.max_retries})",
                            )
                            time.sleep(wait_time)
                            continue

                        # ========== 5xx：服务端错误 ==========
                        if code in (500, 502, 503, 504):
                            # 503 + 近期有限速信号 → 视为 IP 限流
                            if code == 503 and ip_cooldown.has_recent_rate_limit(within_seconds=60.0):
                                cooldown_secs = ip_cooldown.trigger(reason=f"503 during rate-limit storm on key[{key_tag}]")
                                _log(
                                    logger,
                                    "warning",
                                    f"API_KEY[{idx}|{key_tag}] 503+近期限速信号，视为IP限流，"
                                    f"IP全局冷却 {cooldown_secs:.1f}s",
                                )
                                break_to_outer = True
                                break

                            # 正常 5xx：线性退避重试
                            wait_time = cfg.base_delay * attempt_no
                            _log(
                                logger,
                                "warning",
                                f"API_KEY[{idx}|{key_tag}] 服务器错误({code})，"
                                f"等待 {wait_time:.1f}s 后重试 ({attempt_no}/{cfg.max_retries})",
                            )
                            time.sleep(wait_time)
                            continue

                        # ========== 其他未知错误 ==========
                        if real_attempt < cfg.max_retries:
                            wait_time = cfg.base_delay * attempt_no
                            _log(
                                logger,
                                "warning",
                                f"API_KEY[{idx}|{key_tag}] 未知错误({code}: {err_msg[:100]})，"
                                f"等待 {wait_time:.1f}s 后重试 ({attempt_no}/{cfg.max_retries})",
                            )
                            time.sleep(wait_time)
                            continue
                        
                        # 最后一次重试也失败，抛出异常
                        raise

                # 仍在 key 循环内 — break 跳出 for 循环
                if break_to_outer:
                    break

            # key 循环之外 — continue 回到 while 循环顶部（执行 wait_if_needed）
            if break_to_outer:
                continue

        # 超过最大外层尝试次数
        raise RuntimeError(f"超过最大重试次数({max_outer_attempts})，所有 API_KEY 均失败")

    return chat_completion
