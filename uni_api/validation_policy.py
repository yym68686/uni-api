PYDANTIC_ALLOWED_MODULES = (
    "uni_api.config.schema",
    "api route request models",
    "test fixtures",
)

PYDANTIC_HOT_PATH_FORBIDDEN_MODULES = (
    "uni_api.routing",
    "uni_api.upstream",
    "uni_api.rate_limit",
    "uni_api.streaming",
    "provider retry loops",
    "SSE chunk parsers",
)
