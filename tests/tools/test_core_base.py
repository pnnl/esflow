import pytest

from tools.core.base import Param, ToolSpec, esmflow_tool


@pytest.mark.parametrize(
    ("param", "value", "expected"),
    [
        (Param("str"), 12, "12"),
        (Param("int"), "3.7", 3),
        (Param("float"), " 1.25 ", 1.25),
        (Param("bool"), "YeS", True),
        (Param("bool"), "false", False),
        (Param("path"), "a/../b", "a/../b"),
        (Param("list[str]"), "one, two", ["one", "two"]),
        (Param("list[int]"), "2000-2002", [2000, 2001, 2002]),
        (Param("list[int]"), "[2000, 2002]", [2000, 2002]),
        (Param("list[int]"), "-5", [-5]),
        (Param("list[int]"), (1, "2"), [1, 2]),
    ],
)
def test_param_coerce(param, value, expected):
    assert param.coerce(value) == expected


def test_param_coerce_none_uses_default():
    assert Param("int", default=7).coerce(None) == 7
    assert Param("path", default="fallback").coerce("") == "fallback"


def test_tool_spec_parse_config_coerces_and_rejects_unknown_values():
    spec = ToolSpec(
        name="example",
        description="Example tool",
        inputs={"count": Param("int", required=True), "years": Param("list[int]", default=[])},
        outputs={},
    )

    assert spec.parse_config({"count": "2", "years": "2000-2001"}) == {
        "count": 2,
        "years": [2000, 2001],
    }
    assert spec.parse_config({"count": 2, "output_dir": "out", "output_file": "result.csv"}) == {
        "count": 2,
        "years": [],
    }

    with pytest.raises(ValueError, match="Missing required parameter: count"):
        spec.parse_config({})
    with pytest.raises(ValueError, match="Unknown parameter: 'unexpected'"):
        spec.parse_config({"count": 2, "unexpected": True})


def test_decorator_creates_output_dir_and_checks_declared_file_outputs(tmp_path):
    spec = ToolSpec(
        name="writes_csv",
        description="Writes a CSV",
        inputs={"name": Param("str", required=True)},
        outputs={"csv_file": {"type": "csv", "description": "CSV output"}},
    )

    @esmflow_tool(spec)
    def writes_csv(config):
        path = tmp_path / f"{config['name']}.csv"
        path.write_text("value\n1\n")
        return {"csv_file": str(path)}

    result = writes_csv({"name": "result", "output_dir": tmp_path / "nested"})
    assert (tmp_path / "nested").is_dir()
    assert result["csv_file"].endswith("result.csv")
