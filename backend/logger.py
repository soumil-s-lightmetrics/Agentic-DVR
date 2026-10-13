import base64
import re
import logging
from logging.handlers import RotatingFileHandler
import os
from pythonjsonlogger import jsonlogger
from flask import request, has_request_context
import time
from datetime import datetime, timezone
import types

# Constants
LOG_FOLDER = os.path.join(os.getcwd(), os.getenv("LOGS_DIR", "logs"))
ACCESS_LOG_FILE = os.path.join(LOG_FOLDER, 'access.log')
DEBUG_LOG_FILE = os.path.join(LOG_FOLDER, 'debug.log')
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

# Environment variables
LOG_FORMAT = os.getenv("LOG_FORMAT", "human").lower()
LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "true").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
SERVICE_NAME = os.getenv("SERVICE_NAME", "lisa-api")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "unknown")

# Regex patterns
authSchemeRegex = re.compile(r"^[^ ]*")
basicCredentialsRegex = re.compile(r"^basic *([^ ]*)", re.IGNORECASE)

# Sensitive data patterns for redaction
SENSITIVE_PATTERNS = [
    (re.compile(r'"password"\s*:\s*"[^"]*"', re.IGNORECASE), '"password":"***REDACTED***"'),
    (re.compile(r'"api_key"\s*:\s*"[^"]*"', re.IGNORECASE), '"api_key":"***REDACTED***"'),
    (re.compile(r'"token"\s*:\s*"[^"]*"', re.IGNORECASE), '"token":"***REDACTED***"'),
    (re.compile(r'Bearer [A-Za-z0-9\-._~+/]+=*', re.IGNORECASE), 'Bearer ***REDACTED***'),
]

# Paths to skip in debug logging (reduce noise)
SKIP_DEBUG_LOG_PATHS = {'/health-check', '/v1/llm-kb/health-check', '/v2/llm-kb/health-check'}

# Ensure log folder exists
os.makedirs(LOG_FOLDER, exist_ok=True)

# Memoization caches
_logger_cache = {}
_formatter_cache = {}

# Third-party library loggers to configure
THIRD_PARTY_LOGGERS = {
    'httpx': logging.INFO,         # OpenAI SDK HTTP client
    'httpcore': logging.WARNING,   # HTTP core library
    'urllib3': logging.WARNING,    # Requests/boto3 HTTP client
    'boto3': logging.WARNING,      # AWS SDK
    'botocore': logging.WARNING,   # AWS SDK core
    'werkzeug': logging.DEBUG,      # Flask development server
    'openai': logging.WARNING,     # OpenAI SDK
}

def get_env_var(key, default=None):
    """Helper to fetch environment variables with a default value."""
    return os.getenv(key, default)

def configure_third_party_loggers():
    """
    Configure third-party library loggers to reduce noise.

    - Sets appropriate log levels (WARNING by default)
    - Prevents propagation to root logger
    - Optionally redirects to our debug.log
    """
    for logger_name, level in THIRD_PARTY_LOGGERS.items():
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(level)
        lib_logger.propagate = False  # Don't propagate to root logger

        # Optionally add our handler to capture WARNING+ messages
        if not lib_logger.handlers:
            handler = get_log_handler(DEBUG_LOG_FILE, level, get_log_formatter())
            lib_logger.addHandler(handler)

def redact_sensitive_data(message):
    """Redact sensitive data from log messages."""
    if not isinstance(message, str):
        return message

    for pattern, replacement in SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)

    return message

def should_skip_debug_log():
    """Check if current request should skip debug logging."""
    if not has_request_context():
        return False

    return request.path in SKIP_DEBUG_LOG_PATHS

# Custom JSON formatter with auto-injected fields (created once)
class JSONRequestIDFormatter(jsonlogger.JsonFormatter):
    """JSON formatter with auto-injected context fields."""

    def add_fields(self, log_record, record, message_dict):
        try:
            # Auto-inject RequestID if not already present
            if not hasattr(record, 'requestId'):
                if has_request_context():
                    record.requestId = request.environ.get("HTTP_X_REQUEST_ID", "N/A")
                else:
                    record.requestId = "N/A"

            # Call super first to process format string
            super().add_fields(log_record, record, message_dict)

            # Now add/override our custom fields
            log_record['timestamp'] = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
            log_record['logger'] = record.name
            log_record['environment'] = ENVIRONMENT
            log_record['service'] = SERVICE_NAME
            if SERVICE_VERSION != "unknown":
                log_record['version'] = SERVICE_VERSION

            # Redact sensitive data from message
            if 'message' in log_record:
                log_record['message'] = redact_sensitive_data(log_record['message'])

        except Exception as e:
            # Fail gracefully - don't break logging
            log_record['formatter_error'] = str(e)
            super().add_fields(log_record, record, message_dict)

# Custom human-readable formatter with auto-injected RequestID (created once)
class RequestIDFormatter(logging.Formatter):
    """Human-readable formatter with auto-injected RequestID."""

    def format(self, record):
        try:
            # Auto-inject RequestID if not already present
            if not hasattr(record, 'requestId'):
                if has_request_context():
                    record.requestId = request.environ.get("HTTP_X_REQUEST_ID", "N/A")
                else:
                    record.requestId = "N/A"

            # Format the message
            formatted = super().format(record)

            # Redact sensitive data
            return redact_sensitive_data(formatted)
        except Exception as e:
            # Fail gracefully - return basic format
            return f"{record.levelname}: {record.getMessage()} [formatter_error: {e}]"

def get_log_formatter(format_type="default"):
    """
    Returns the appropriate log formatter (cached for performance).

    Supports hybrid logging:
    - JSON format for production (set LOG_FORMAT=json)
    - Human-readable format for development (set LOG_FORMAT=human or leave unset)
    """
    # Check cache first
    cache_key = f"{format_type}_{LOG_FORMAT}"
    if cache_key in _formatter_cache:
        return _formatter_cache[cache_key]

    # Access logs are always JSON (for parsing in log aggregation systems)
    if format_type == "access":
        formatter = JSONRequestIDFormatter(
            '%(name)s %(levelname)s %(requestId)s %(reqTimestamp)s %(httpVersion)s %(path)s %(route)s %(queryString)s '
            '%(method)s %(authScheme)s %(forwardedFor)s %(userAgent)s %(reqContentType)s %(referer)s %(origin)s %(host)s '
            '%(visitId)s %(accessorId)s %(accountType)s %(clientId)s '
            '%(resTimestamp)s %(responseTimeMs)s %(statusCode)s %(resContentLength)s %(resContentType)s '
            '%(errMessage)s %(errStack)s'
        )
    elif LOG_FORMAT == "json":
        # Structured JSON for production (easy parsing in CloudWatch, Datadog, etc.)
        formatter = JSONRequestIDFormatter(
            '%(timestamp)s %(logger)s %(levelname)s %(requestId)s %(message)s'
        )
    else:
        # Human-readable format for local development
        formatter = RequestIDFormatter(
            '%(asctime)s | %(name)s | %(levelname)s | RequestID: %(requestId)s | %(message)s'
        )

    # Cache and return
    _formatter_cache[cache_key] = formatter
    return formatter

class HealthCheckFilter(logging.Filter):
    """Filter to skip health check logs in debug logger."""

    def filter(self, record):
        # Allow all logs outside request context
        if not has_request_context():
            return True

        # Skip health check paths
        return request.path not in SKIP_DEBUG_LOG_PATHS

def get_log_handler(filename, level, formatter):
    """Creates and returns a RotatingFileHandler."""
    handler = RotatingFileHandler(
        filename, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler

def configure_logger(logger_name, handlers, level=logging.DEBUG, add_health_check_filter=False):
    """Configures and returns a logger with the specified handlers."""
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger  # Avoid adding duplicate handlers

    logger.setLevel(level)
    logger.propagate = False  # Prevent propagation to root logger

    # Add health check filter for debug logger
    if add_health_check_filter:
        logger.addFilter(HealthCheckFilter())

    for handler in handlers:
        logger.addHandler(handler)

    # Add stream handler if LOG_TO_STDOUT is enabled
    if LOG_TO_STDOUT:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(get_log_formatter())
        logger.addHandler(stream_handler)

    return logger

def access_logger():
    """Returns a logger for access logs"""
    if 'access_logger' in _logger_cache:
        return _logger_cache['access_logger']

    access_handler = get_log_handler(
        ACCESS_LOG_FILE, logging.INFO, get_log_formatter("access")
    )
    logger = configure_logger('access_logger', [access_handler])
    _logger_cache['access_logger'] = logger
    return logger

def debug_logger():
    """Returns a logger for debug logs with health check filtering"""
    if 'debug_logger' in _logger_cache:
        return _logger_cache['debug_logger']

    debug_handler = get_log_handler(
        DEBUG_LOG_FILE, logging.DEBUG, get_log_formatter()
    )
    logger = configure_logger('debug_logger', [debug_handler], add_health_check_filter=True)
    _logger_cache['debug_logger'] = logger
    return logger

def log_request(response, logger, total_bytes=0, ttfb_ms=None):
    """Logs request and response details, safely handling streaming responses."""
    try:
        # Skip logging for OPTIONS requests and ELB health checks early for efficiency
        user_agent = request.headers.get('User-Agent')
        if request.method == 'OPTIONS' or (user_agent and user_agent.startswith('ELB-HealthChecker')):
            return

        status_code = getattr(response, 'status_code', 200)
        authHeader = request.headers.get('Authorization')
        authScheme = request.headers.get("Authorization", "none").split()[0] if "Authorization" in request.headers else "none"

        is_streaming = isinstance(response.response, (types.GeneratorType, types.AsyncGeneratorType))

        # Get response body only if it's safe
        if not is_streaming and hasattr(response, "get_data"):
            try:
                response_body = response.get_data(as_text=True)
                res_content_length = len(response_body)
            except RuntimeError:
                # Response is in direct passthrough mode (e.g., send_file)
                res_content_length = total_bytes
        else:
            res_content_length = total_bytes

        # Determine response content type
        res_content_type = response.headers.get("Content-Type", "unknown")
        response_time = round((time.time() - getattr(request, 'start_time', time.time())) * 1000, 2)

        clientId = (
            request.headers.get('x-lm-client-id') or
            getattr(getattr(request, '_lmClient', None), 'clientId', None) or
            (
                authScheme.lower() == 'basic' and
                authHeader and basicCredentialsRegex.match(authHeader) and
                base64.b64decode(basicCredentialsRegex.match(authHeader).group(1)).decode('utf-8').split(':', 1)[0]
            )
        ) or "Unknown"

        log_data = {
            "visitId": request.headers.get('x-visit-id', 'Unknown'),
            "accessorId": request.headers.get('x-lm-accessor-id', 'Unknown'),
            "accountType": request.headers.get('x-lm-account-type', 'Unknown'),
            "clientId": clientId,
            "requestId": request.environ.get("HTTP_X_REQUEST_ID"),
            "reqTimestamp": datetime.fromtimestamp(getattr(request, 'start_time', time.time()), timezone.utc).isoformat(),
            "httpVersion": request.environ.get("SERVER_PROTOCOL", "HTTP/1.1"),
            "method": request.method,
            "path": request.path,
            "route": getattr(request.url_rule, 'rule', "unknown"),
            "queryString": request.query_string.decode("utf-8"),
            "authScheme": authScheme,
            "forwardedFor": request.headers.get("X-Forwarded-For", request.remote_addr),
            "userAgent": request.headers.get("User-Agent", "Unknown"),
            "referer": request.headers.get("Referer", "Unknown"),
            "origin": request.headers.get("Origin", "Unknown"),
            "host": request.headers.get("Host", "Unknown"),
            "reqContentType": request.headers.get("Content-Type", "Unknown"),
            "resTimestamp": datetime.now(timezone.utc).isoformat(),
            "responseTimeMs": response_time,
            "ttfbMs": ttfb_ms if ttfb_ms is not None else response_time,
            "statusCode": status_code,
            "resContentLength": res_content_length,
            "resContentType": res_content_type,
            "errMessage": getattr(response, "message", None),
            "errStack": getattr(response, "stack", None),
        }

        logger.info(
            f"Request completed with status code {status_code} | Path: {request.path}",
            extra=log_data
        )

    except Exception as e:
        logger.error(f"Failed to log request: {e}", exc_info=True)


def websocket_logger():
    """Returns a dedicated logger for WebSocket connections."""
    if 'websocket_logger' in _logger_cache:
        return _logger_cache['websocket_logger']

    WEBSOCKET_LOG_FILE = os.path.join(LOG_FOLDER, 'websocket.log')
    
    websocket_handler = get_log_handler(
        WEBSOCKET_LOG_FILE, logging.INFO, get_log_formatter()
    )
    
    logger = configure_logger('websocket_logger', [websocket_handler])
    _logger_cache['websocket_logger'] = logger
    return logger

configure_third_party_loggers()

