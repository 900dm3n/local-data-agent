"""Strict, deliberately small execution plan vocabulary."""
from typing import Any, Literal
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

class PlanStep(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    step_id: str
    type: Literal['LOAD', 'SELECT', 'RUN', 'SAVE']
    output_name: str
    depends_on: list[str] = Field(default_factory=list)
    source_id: str | None = None
    table: str | None = None
    sql: str | None = None
    operator: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check(self):
        for value in [self.step_id, self.output_name, *self.depends_on]:
            if not IDENTIFIER.fullmatch(value):
                raise ValueError('Step, dependency and output names must be simple identifiers')
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError('Duplicate dependencies')
        if self.sql and len(self.sql) > 32768:
            raise ValueError('SQL exceeds 32768 characters')
        if self.type == 'LOAD':
            if not self.source_id or not self.table or self.depends_on or self.operator:
                raise ValueError('LOAD requires source_id/table, without dependencies/operator')
        elif self.type == 'SELECT':
            if not self.sql or not self.depends_on or self.source_id or self.table or self.operator:
                raise ValueError('SELECT requires SQL and dependencies, without source/operator')
        elif self.type == 'RUN':
            if not self.operator or not self.depends_on or self.sql or self.source_id or self.table:
                raise ValueError('RUN requires a registered operator and dependencies')
        elif self.type == 'SAVE':
            if len(self.depends_on) != 1 or self.sql or self.source_id or self.table or self.operator:
                raise ValueError('SAVE requires exactly one dependency and no SQL/source/operator')
        return self
