from uni_api.validation_policy import (
    PYDANTIC_ALLOWED_MODULES,
    PYDANTIC_HOT_PATH_FORBIDDEN_MODULES,
)


def test_validation_policy_documents_pydantic_boundaries():
    assert "uni_api.config.schema" in PYDANTIC_ALLOWED_MODULES
    assert "uni_api.routing" in PYDANTIC_HOT_PATH_FORBIDDEN_MODULES
    assert "SSE chunk parsers" in PYDANTIC_HOT_PATH_FORBIDDEN_MODULES
