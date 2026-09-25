"""Example 'user code' brought into ESMFlow by the onboarding agent.

Every module here is deliberately plain science code: no ESMFlow imports, no
knowledge of ``ToolSpec``, no output-directory conventions. The onboarding agent
introspects these functions and generates thin adapters under
``tools/<category>/`` that satisfy the ESMFlow tool contract on their behalf.
"""
